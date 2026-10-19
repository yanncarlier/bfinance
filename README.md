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
4. Create a `.env` file with your API credentials
5. Run in paper mode first: `PAPER_TRADING=true uv run ma_crossover_bot.py`

Useful links and editor setup:

- ALL You Need To KNOW About Moving Averages: https://www.youtube.com/watch?v=2mXduyOuzgU
- Deposit crypto (Binance): https://www.binance.com/en/my/wallet/account/main/deposit/crypto/BTC
- Open in VS Code: `code .`, then `Shift + Ctrl + P` for the command palette.

## Project Structure

```
bfinance/
├── main.py                     # Main application entry point
├── ma_crossover_bot.py         # Moving average crossover strategy
├── ma_crossover_bot_rt.py      # MA crossover bot (real-time variant)
├── ma_crossover_bot_rt_tb.py   # MA crossover bot (real-time, TradeBot variant)
├── backtest.py                 # Backtesting framework
├── backtest_2.py               # Backtesting experiments
├── backtest_CT_loop.py         # Backtest with CT loop
├── backtest_CT_loop_tb.py      # Backtest with CT loop (TradeBot variant)
├── backtest_tb.py              # Backtest (TradeBot variant)
├── helper_functions.py         # Shared utility functions
├── simple_candle_plot.py       # Candlestick chart plotting
├── yahoo_price_tracker.py      # Yahoo Finance price tracking
├── latency.sh                  # Latency measurement script
├── requirements.txt            # Dependencies (legacy)
├── pyproject.toml              # Project configuration
├── uv.lock                     # Lock file for uv
├── .env                        # Environment variables (not committed)
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## Build / Lint / Test Commands

### Package Management

- **Install dependencies**: `uv sync`
- **Add dependency**: `uv add <package>`
- **Remove dependency**: `uv remove <package>`
- **Update dependencies**: `uv sync --upgrade`
- **Install from requirements.txt**: `uv pip install -r requirements.txt`

### Linting and Code Quality

- **Format code**: `ruff format .` (install with `uv add ruff`)
- **Lint code**: `ruff check .` (install with `uv add ruff`)
- **Fix linting issues**: `ruff check --fix .`
- **Type checking**: `mypy .` (install with `uv add mypy`)

### Testing

- **Run all tests**: `python -m pytest` (install with `uv add pytest`)
- **Run specific test**: `python -m pytest tests/test_file.py::test_function`
- **Run tests with coverage**: `python -m pytest --cov=.` (install with `uv add pytest-cov`)

### Running Bots

- **Main application**: `uv run main.py`
- **MA Crossover Bot**: `uv run ma_crossover_bot.py`
- **Backtest**: `uv run backtest.py`

## Configuration

- **Environment Variables**: Use a `.env` file for API keys and secrets.
- **Configuration Dictionaries**: Use uppercase dictionaries for trading parameters.
- **Validation**: Always validate required environment variables at startup.

```python
# Load and validate environment
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
if not API_KEY or not API_SECRET:
    raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
```

## Code Style Guidelines

### Python Standards

- **PEP 8 Compliance**: Follow PEP 8 style guidelines.
- **Line Length**: Maximum 88 characters (Black/Ruff default).
- **Imports**: Use absolute imports, grouped by standard library, third-party, then local.
- **Type Hints**: Use type hints for function parameters and return values.
- **Docstrings**: Use Google-style docstrings for all public functions.

### Import Organization

```python
# Standard library imports
import logging
import time
from datetime import datetime

# Third-party imports
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Local imports
import helper_functions
import backtest
```

### Naming Conventions

- **Functions**: `snake_case` (e.g., `calculate_position_size`)
- **Variables**: `snake_case` (e.g., `current_price`, `position_size`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `API_KEY`, `MAX_RETRIES`)
- **Classes**: `PascalCase` (if any classes are added)
- **Modules**: `snake_case` (e.g., `helper_functions.py`)

### Logging Standards

- **Logger Setup**: Use a consistent logging format with timestamps.
- **Log Levels**: INFO for normal operations, WARNING for issues, ERROR for failures.
- **File and Console**: Log to both file and console for debugging.

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### Error Handling

- **Try/Except Blocks**: Use specific exception types when possible.
- **Graceful Degradation**: Continue operation when possible, log errors.
- **API Failures**: Implement retry logic with exponential backoff.
- **Resource Cleanup**: Use context managers for file/database connections.

```python
def fetch_ohlcv(symbol, timeframe, limit, retries=3):
    for attempt in range(retries):
        try:
            # API call here
            return df
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None
```

### Data Structures

- **Dictionaries for Config**: Use nested dictionaries for configuration.
- **DataFrames for OHLCV**: Use pandas DataFrames for market data.
- **Named Tuples**: Consider namedtuples for complex return values.

### Trading Logic Patterns

- **Position Tracking**: Use a dictionary to track position state.
- **Risk Management**: Always calculate position size based on risk parameters.
- **Fee Calculation**: Account for trading fees in P/L calculations.
- **Validation**: Validate inputs before executing trades.

### Security Practices

- **API Keys**: Never commit API keys or secrets to the repository.
- **Paper Trading**: Always support paper trading mode for testing.
- **Input Validation**: Validate all user inputs and API responses.
- **Rate Limiting**: Respect exchange rate limits to avoid bans.

### Testing Guidelines

- **Unit Tests**: Test individual functions with mock data.
- **Integration Tests**: Test complete trading workflows.
- **Mock Exchanges**: Use CCXT test exchanges for integration testing.
- **Edge Cases**: Test with extreme market conditions.

### Documentation

- **README Updates**: Update README.md for new features.
- **Function Comments**: Document complex logic with inline comments.
- **Configuration**: Document all configuration parameters.

## Performance Considerations

- **API Rate Limits**: Respect exchange rate limits (Binance: ~1200 req/min).
- **Memory Usage**: Be mindful of DataFrame sizes for long backtests.
- **Network Efficiency**: Cache balance fetches and minimize API calls.
- **Logging Overhead**: Use appropriate log levels to avoid performance impact.

## Risk Management

- **Position Sizing**: Never risk more than the specified percentage per trade.
- **Stop Losses**: Always implement trailing or fixed stop losses.
- **Paper Trading**: Test all changes in paper trading mode first.
- **Backtesting**: Validate strategies with historical data before live trading.

## Development Workflow

### Feature Development

1. Create a feature branch from main
2. Implement changes following style guidelines
3. Run linting and formatting: `ruff check --fix . && ruff format .`
4. Test changes with paper trading mode
5. Run backtests to validate strategy
6. Update documentation
7. Commit with a descriptive message

### Code Review Checklist

- [ ] Follows code style guidelines
- [ ] Includes appropriate error handling
- [ ] Uses logging for important operations
- [ ] Validates inputs and handles edge cases
- [ ] Includes docstrings for public functions
- [ ] Tested with paper trading mode
- [ ] Backtested for performance validation
