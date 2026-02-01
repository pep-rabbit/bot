import asyncio
from pprint import pprint
from typing import Literal, overload

import uvloop
from .client import Client
import msgspec


class ResServerTime(msgspec.Struct):
    timeSecond: str
    timeNano: str


class PriceKline(msgspec.Struct, array_like=True):
    startTime: str
    openPrice: str
    highPrice: str
    lowPrice: str
    closePrice: str


class ResPriceKline(msgspec.Struct):
    symbol: str
    category: str
    item: list[PriceKline] = msgspec.field(name="list")


class Ticker(msgspec.Struct):
    symbol: str
    turnover24h: str


class ResTicker(msgspec.Struct):
    category: str
    item: list[Ticker] = msgspec.field(name="list")


class API(Client):
    async def server_time(self):
        ans = await self.request("GET", "/v5/market/time")
        return msgspec.json.decode(ans.result, type=ResServerTime)

    async def mark_price_kline(
        self,
        category: Literal["linear", "inverse"],
        symbol: str = None,
        interval: Literal[
            "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "M", "W"
        ] = None,
        start: int = None,
        end: int = None,
        limit: int = 200,
    ):
        ans = await self.request(
            "GET",
            "/v5/market/kline",
            params={
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )

        return msgspec.json.decode(ans.result, type=ResPriceKline)

    async def tickers(self, category: Literal["linear", "inverse"]):
        ans = await self.request(
            "GET",
            "/v5/market/tickers",
            params={"category": category},
        )
        return msgspec.json.decode(ans.result, type=ResTicker)


async def main():
    conn = API(
        "HRGA0f12NYcrxBfRyD",
        "KvXnDCfMJj5DLSZhniJdAgHcABNarlXZ2wKl",
        "https://api.bybit.com",
    )
    tickers = await conn.tickers("linear")
    pprint(tickers.item)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=uvloop.Loop)
