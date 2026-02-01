import asyncio
import msgspec
import polars as pl
import uvloop
from bybit.api import API
from datetime import datetime, timezone, timedelta

# --- ЛОГИКА СТРАТЕГИИ ---


def filter_by_trend(lf: pl.LazyFrame, high_tf_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Определяет тренд на старшем ТФ (1 час).
    Используем join_asof, чтобы каждая 15-минутная свеча знала,
    какой был тренд на часовике в этот момент.
    """
    trend_lf = (
        high_tf_lf.with_columns(trend_sma=pl.col("close").rolling_mean(window_size=50))
        .with_columns(is_uptrend=(pl.col("close") > pl.col("trend_sma")))
        .select(["timestamp", "is_uptrend"])
    )

    return lf.sort("timestamp").join_asof(
        trend_lf.sort("timestamp"), on="timestamp", strategy="backward"
    )


def find_levels_and_touches(
    lf: pl.LazyFrame, proximity_pct: float = 0.003
) -> pl.LazyFrame:
    """
    Ищет уровни и касания.
    Для 15м ТФ окно поиска уровня 20 свечей = 5 часов (локальная полка).
    """
    return (
        lf.with_columns(
            [
                pl.col("high").rolling_max(window_size=20).alias("resistance_level"),
                pl.col("low").rolling_min(window_size=20).alias("accumulation_base"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("resistance_level") - pl.col("high")).abs()
                    / pl.col("resistance_level")
                    < proximity_pct
                )
                .cast(pl.Int32)
                .alias("is_touch")
            ]
        )
        .with_columns(
            [pl.col("is_touch").rolling_sum(window_size=20).alias("touches_count")]
        )
    )


def check_accumulation_density(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        volatility=(pl.col("high") - pl.col("low")) / pl.col("close")
    ).with_columns(is_dense_accumulation=(pl.col("volatility") < 0.015))


def is_extremum_base(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        is_local_extreme=(
            pl.col("resistance_level") == pl.col("high").rolling_max(window_size=100)
        )
    )


def calculate_trade_params(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.with_columns(
            [
                (pl.col("resistance_level") - pl.col("accumulation_base")).alias(
                    "pattern_height"
                )
            ]
        )
        .with_columns(
            [
                (pl.col("resistance_level") + pl.col("pattern_height")).alias(
                    "take_profit"
                ),
                pl.col("accumulation_base").alias("stop_loss"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("take_profit") - pl.col("close"))
                    / (pl.col("close") - pl.col("stop_loss"))
                ).alias("risk_reward")
            ]
        )
    )


def get_strategy_signals(raw_data: pl.LazyFrame, h1_data: pl.LazyFrame):
    pipeline = (
        raw_data.pipe(filter_by_trend, high_tf_lf=h1_data)
        .pipe(find_levels_and_touches)
        .pipe(check_accumulation_density)
        .pipe(is_extremum_base)
        .pipe(calculate_trade_params)
    )

    # Фильтр: Минимум 3 из 4 условий + RR >= 3
    return (
        pipeline.filter(
            (
                pl.col("is_uptrend").cast(pl.Int8)
                + (pl.col("touches_count") >= 3).cast(pl.Int8)
                + pl.col("is_dense_accumulation").cast(pl.Int8)
                + pl.col("is_local_extreme").cast(pl.Int8)
            )
            >= 3
        ).filter(pl.col("risk_reward") >= 3)
    ).collect()


# --- РАБОТА С API ---


async def get_bybit_data(
    conn: API, symbol: str, interval: str, limit: int = 200
) -> pl.LazyFrame:
    try:
        tickers = await conn.mark_price_kline(
            "linear", symbol=symbol, interval=interval, limit=limit
        )
        if not tickers.item:
            return None

        return (
            pl.DataFrame(
                msgspec.to_builtins(tickers.item),
                schema=["timestamp", "open", "high", "low", "close"],
                orient="row",
            )
            .lazy()
            .with_columns(
                [
                    pl.col("timestamp").cast(pl.Int64),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                ]
            )
            .sort("timestamp")
        )
    except Exception:
        return None


async def analyze_ticker(sem, conn, symbol):
    """Анализирует монету на 15м ТФ."""
    async with sem:
        try:
            # ИЗМЕНЕНИЕ: Запрашиваем 15 минут (рабочий) и 60 минут (тренд)
            raw_task = get_bybit_data(conn, symbol, "15", limit=500)
            h1_task = get_bybit_data(conn, symbol, "60", limit=500)

            raw_data, h1_data = await asyncio.gather(raw_task, h1_task)

            if raw_data is None or h1_data is None:
                return None

            signals = get_strategy_signals(raw_data, h1_data)

            if signals.is_empty():
                return None

            # Проверяем свежесть
            last_candle = signals.tail(1)
            last_ts = last_candle["timestamp"].item(0)
            current_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

            # ИЗМЕНЕНИЕ: Сигнал актуален в течение 30 минут (2 свечи по 15м)
            if current_ts - last_ts < 30 * 60 * 1000:

                close_price = last_candle["close"].item(0)
                level = last_candle["resistance_level"].item(0)
                dist_pct = abs(level - close_price) / level * 100

                return {
                    "coin": symbol,
                    "type": "LONG",
                    "level": level,
                    "touches": last_candle["touches_count"].item(0),
                    "trend": (
                        "вверх" if last_candle["is_uptrend"].item(0) else "флэт/вниз"
                    ),
                    "volume": "высокий",
                    "distance": round(dist_pct, 2),
                }

            return None

        except Exception:
            return None


async def scan_market_v2(conn: API):
    try:
        # Используем tickers (стандартное название метода)
        resp = await conn.tickers(category="linear")
        all_tickers = resp.item

        usdt_tickers = [
            t
            for t in all_tickers
            if t.symbol.endswith("USDT") and float(t.turnover24h) > 10_000_000
        ]
        usdt_tickers.sort(key=lambda x: float(x.turnover24h), reverse=True)
        # Сканируем топ-50
        top_coins = [t.symbol for t in usdt_tickers]

        sem = asyncio.Semaphore(10)
        tasks = [analyze_ticker(sem, conn, symbol) for symbol in top_coins]
        results = await asyncio.gather(*tasks)

        return [res for res in results if res is not None]
    except Exception as e:
        print(f"⚠️ Ошибка получения тикеров: {e}")
        return []


# --- MAIN ---


async def main():
    print("🚀 Запуск сканера [Стратегия 4.8 | TF: 15m | Trend: 1h]")

    conn = API(
        "HRGA0f12NYcrxBfRyD",
        "KvXnDCfMJj5DLSZhniJdAgHcABNarlXZ2wKl",
        "https://api.bybit.com",
    )

    seen_signals = {}
    COOLDOWN_MINUTES = 60  # Не показывать повторно 1 час
    iteration = 1

    try:
        while True:
            start_time = datetime.now()
            print(f"\n🔍 [Итерация {iteration}] Сканирую рынок...")

            raw_results = await scan_market_v2(conn)
            new_unique_signals = []
            current_time = datetime.now()

            # Фильтрация дублей
            for res in raw_results:
                symbol = res["coin"]
                if symbol not in seen_signals or (
                    current_time - seen_signals[symbol]
                ) > timedelta(minutes=COOLDOWN_MINUTES):
                    new_unique_signals.append(res)
                    seen_signals[symbol] = current_time

            # Вывод
            if new_unique_signals:
                print(f"\n🔥 НОВЫХ СИГНАЛОВ: {len(new_unique_signals)}")
                print("=" * 30)
                for res in new_unique_signals:
                    print(f"Монета: {res['coin']}")
                    print(f"Тип: {res['type']}")
                    print(f"Уровень: {res['level']}")
                    print(f"Касаний: {res['touches']}")
                    print(f"Тренд: {res['trend']}")
                    print(f"Объём: {res['volume']}")
                    print(f"Расстояние до цены: {res['distance']}%")
                    print("-" * 30)
            else:
                elapsed = (datetime.now() - start_time).total_seconds()
                hidden_str = ", ".join([f"{k}" for k in seen_signals.keys()])
                if len(raw_results) > 0:
                    print(f"😴 Новых нет. Скрыты повторы: {hidden_str}")
                else:
                    print(f"😴 Сигналов нет. (Scan time: {elapsed:.1f}s)")

            # Очистка памяти
            keys_to_remove = [
                k
                for k, v in seen_signals.items()
                if (current_time - v) > timedelta(hours=2)
            ]
            for k in keys_to_remove:
                del seen_signals[k]

            iteration += 1
            await asyncio.sleep(20)

    except KeyboardInterrupt:
        print("\n🛑 Стоп.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=uvloop.Loop)
