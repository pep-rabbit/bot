import asyncio
import hashlib
import hmac
import time
from typing import Literal
import aiohttp
import msgspec
import orjson
from yarl import URL


class Client:
    class Response(msgspec.Struct):
        retCode: int
        retMsg: str
        result: msgspec.Raw
        retExtInfo: msgspec.Raw
        time: int

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bybit.com",
        limit: int = 100,
    ):
        self._api_key = api_key
        self._recv_window = "5000"

        self._signature = hmac.new(api_secret.encode("utf-8"), digestmod=hashlib.sha256)
        self._decoder = msgspec.json.Decoder(self.Response)

        self.semaphore = asyncio.Semaphore(value=limit)
        connector = aiohttp.TCPConnector(limit=limit)
        timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=10)
        self._session = aiohttp.ClientSession(
            base_url=base_url, connector=connector, timeout=timeout
        )

    async def close(self):
        if not self._session.closed:
            await self._session.close()

    def _build_headers(self, payload_bytes: bytes = None):
        timestamp = str(time.time_ns() // 1_000_000)

        signature = self._signature.copy()
        signature.update(
            (timestamp + self._api_key + self._recv_window).encode("utf-8")
            + payload_bytes
            or b""
        )

        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-SIGN": signature.hexdigest(),
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self._recv_window,
        }

    async def request(
        self,
        method: Literal["GET", "POST"],
        url: str,
        *,
        params: dict | None = None,
    ):
        async with self.semaphore:
            data = None
            if method == "GET":
                haders = self._build_headers(
                    URL.build(query=params).query_string.encode()
                )
            elif method == "POST":
                data = orjson.dumps(params)
                haders = self._build_headers(data)

            async with self._session.request(
                method, url, headers=haders, params=params, data=data
            ) as response:
                raw_body = await response.read()
                return self._decoder.decode(raw_body)
