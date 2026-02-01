FROM python:3.14-alpine
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --compile-bytecode --locked

COPY ./main.py ./main.py
COPY ./bybit ./bybit

CMD ["uv", "run", "main.py"]