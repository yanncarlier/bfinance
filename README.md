# bfinance Trading Bots

## What This Program Does

This project is a set of tools for experimenting with automated Bitcoin trading on Binance. At its core is a **moving-average (MA) crossover bot**: it watches the BTC/USDT price and compares a fast-moving average with a slow one.

- When the fast average crosses **above** the slow one (a "golden cross"), the bot **buys**.
- When the fast average crosses **below** the slow one (a "death cross"), the bot **sells**.

While it holds a position, it also protects it with a **trailing stop-loss** that sells automatically if the price drops far enough from its high point. Every trade's profit/loss is reported with fees included, so you see realistic results rather than optimistic ones.

Before trading, it can **backtest**: it downloads historical price data and automatically tries many combinations of fast/slow averages, then uses the best-performing pair for live trading. This means you don't have to guess which numbers to use — the bot tests them first.

The main things it can do:

- **Trade automatically** on BTC/USDT using MA crossover signals, with position sizing, a max trade size, and a trailing stop-loss.
- **Paper trade safely** — by default, no real money moves; orders are only simulated and logged. Real trading requires a manual confirmation and an explicit setting.
- **Backtest strategies** over historical data, including simulated fees and slippage, to check performance before risking funds.
- **Plot and inspect prices** — save candlestick charts and track BTC prices from Yahoo Finance or Binance.
- **Respect exchange limits** — built-in rate limiting, cached balance lookups, and retry-with-backoff on failed API calls.

It's meant for learning and experimenting with trading strategies, **not** as a proven money-maker — see the disclaimer below.

## Overview

Cryptocurrency trading bots built in Python 3.12+ using [CCXT](https://github.com/ccxt/ccxt) for exchange access and [uv](https://docs.astral.sh/uv/) for package management. The project focuses on moving-average crossover strategies, with backtesting, live/paper trading, and price-tracking tooling.

**Disclaimer: Use at Your Own Risk**
This is experimental beta software and may contain bugs or exhibit unexpected behavior. It is provided "as-is" without any warranties. Use of this software is entirely at your own discretion and risk.

## Getting Started

```bash
git clone git@github.com:yanncarlier/bfinance.git
cd bfinance/
uv run main.py
source .venv/bin/activate
uv pip install -r requirements.txt --reinstall
uv pip freeze > requirements.txt
uv run ma_crossover_bot.py
```

Quick start with uv:

1. Clone repository: `git clone <repo-url>`
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install dependencies: `uv sync`
4. Copy the config template and add your credentials: `cp .env.example .env`
5. Run in paper mode first: `PAPER_TRADING=true uv run ma_crossover_bot.py`

## Configuration

All parameters are read from environment variables by `config.py`. The values
that used to be hardcoded are kept as defaults, so behaviour is unchanged if
`.env` is left empty.

- Secrets (`API_KEY`, `API_SECRET`) live only in `.env`, which is git-ignored.
- Shared settings have no prefix: `SYMBOL`, `TAKER_FEE_PCT`, `SLIPPAGE_PCT`,
  `PAPER_TRADING`, `BALANCE_CACHE_REFRESH_SEC`, ...
- Script-specific settings are prefixed by script: `MA_` (`ma_crossover_bot.py`),
  `RT_` / `RT_TB_` (real-time bots), and `BT_` / `BT_TB_` / `BT_CT_` / `BT_CT_TB_`
  (backtest scripts). See `.env.example` for the full list.

