# scripts/

Optional, standalone tools and experiments. **None of these are needed to run
the main bot** (`../ma_crossover_bot.py`); they are kept here so the repo root
only contains the core: `ma_crossover_bot.py`, `backtest.py`, `config.py`.

Every script is run from the **repository root** so that imports and any
relative paths resolve consistently:

```bash
uv run scripts/<script>.py
```

The scripts that need configuration import `../config.py`, which loads the
repo-root `.env`. They add the repo root to `sys.path` themselves, so they work
when run directly. Remember: with `PAPER_TRADING=false` the real-time bots place
**real orders**.

Common variables (see `../.env.example`): `API_KEY`, `API_SECRET`,
`PAPER_TRADING`, `SYMBOL`, `BASE_CURRENCY`, `QUOTE_CURRENCY`, `TAKER_FEE_PCT`,
`SLIPPAGE_PCT`, `BALANCE_CACHE_REFRESH_SEC`, plus the shared backtest keys
`INIT_USDT`, `POSITION_SIZE_PCT`, `MAX_TRADE_USD`, `TAKE_PROFIT_PCT`,
`TRAILING_STOP_PCT`, `TOP_COMBOS_TO_DISPLAY`. All are optional; anything unset
falls back to the default in the code.

---

## `ma_crossover_bot_rt.py`

Real-time MA-crossover bot (async), long-only spot, with ADX/RSI/volume
filters, take-profit and trailing stop.

```bash
PAPER_TRADING=true uv run scripts/ma_crossover_bot_rt.py
```

Env vars (`RT_`): `RT_CANDLE_TIMEFRAME`, `RT_SHORT_WINDOW`, `RT_LONG_WINDOW`,
`RT_ADX_THRESHOLD`, `RT_POSITION_SIZE_PCT`, `RT_MAX_TRADE_USD`,
`RT_TAKE_PROFIT_PCT`, `RT_TRAILING_STOP_PCT`, `RT_ADX_PERIOD`, `RT_RSI_PERIOD`,
`RT_RSI_OVERBOUGHT`, `RT_CANDLE_POLLER`, `RT_VOLUME_MA_PERIOD`,
`RT_VOLUME_MULTIPLIER`. Logs to `spot_ma_bot.log` (in the current directory).

## `ma_crossover_bot_rt_tb.py`

Same as above plus a historical date-range preload on start.

```bash
PAPER_TRADING=true uv run scripts/ma_crossover_bot_rt_tb.py
```

Env vars (`RT_TB_`): all of the `RT_` set above (with the `RT_TB_` prefix),
plus `RT_TB_START_DATE` and `RT_TB_END_DATE` (blank end date = today).

## `backtest_tb.py`

Single backtest over a fixed date range (the "TradeBot" variant).

```bash
uv run scripts/backtest_tb.py
```

Env vars (`BT_TB_`): `BT_TB_CANDLE_TIMEFRAME`, `BT_TB_START_DATE`,
`BT_TB_END_DATE` (blank = today). Strategy sizing/stops come from the shared
keys above.

## `backtest_CT_loop.py`

Backtest that loops over multiple timeframes ("CT loop").

```bash
uv run scripts/backtest_CT_loop.py
```

Env vars (`BT_CT_`): `BT_CT_CANDLE_TIMEFRAME`, `BT_CT_DATA_LIMIT`.

## `backtest_CT_loop_tb.py`

CT loop across timeframes over a fixed date range.

```bash
uv run scripts/backtest_CT_loop_tb.py
```

Env vars (`BT_CT_TB_`): `BT_CT_TB_CANDLE_TIMEFRAME`, `BT_CT_TB_START_DATE`,
`BT_CT_TB_END_DATE` (blank = today).

## `backtest_2.py`

Throwaway experiment: reads `BTC-USD_daily.csv` from the current directory and
prints moving averages. That CSV is **not** in the repo, so this only runs if
you supply one. Uses no environment config.

```bash
uv run scripts/backtest_2.py
```

## `helper_functions.py`

Small Binance helpers (`print_balance`, `get_last_price`, `place_market_order`).
Reads `API_KEY` / `API_SECRET` straight from `.env` (it predates `config.py`).
Running it fetches and prints your balance and the last price; the order calls
are commented out inside.

```bash
uv run scripts/helper_functions.py
```

## `simple_candle_plot.py`

Fetches recent BTC/USDT candles and saves a candlestick chart (matplotlib /
mplfinance). Symbol, timeframe and window are set at the top of the file.

```bash
uv run scripts/simple_candle_plot.py
```

## `yahoo_price_tracker.py`

Tracks the BTC-USD price from Yahoo Finance and prints 24h change. Uses
`yfinance`; no API keys or env config.

```bash
uv run scripts/yahoo_price_tracker.py
```

## `main.py`

Placeholder entry point that just prints a greeting. Not used by anything.

```bash
uv run scripts/main.py
```
