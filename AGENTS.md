# AGENTS.md - Development Guidelines for bfinance Trading Bots

## Overview
This document provides guidelines for coding assistants working on the bfinance Python trading bot project. The project uses Python 3.12+ with uv package management and focuses on cryptocurrency trading strategies using CCXT library.

## Build/Lint/Test Commands

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

## Code Style Guidelines

### Python Standards
- **PEP 8 Compliance**: Follow PEP 8 style guidelines
- **Line Length**: Maximum 88 characters (Black/Ruff default)
- **Imports**: Use absolute imports, group by standard library, third-party, local
- **Type Hints**: Use type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings for all public functions

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

### Configuration Management
- **Environment Variables**: Use `.env` file for API keys and secrets
- **Configuration Dictionaries**: Use uppercase dictionaries for trading parameters
- **Validation**: Always validate required environment variables at startup

```python
# Load and validate environment
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
if not API_KEY or not API_SECRET:
    raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
```

### Logging Standards
- **Logger Setup**: Use consistent logging format with timestamps
- **Log Levels**: INFO for normal operations, WARNING for issues, ERROR for failures
- **File and Console**: Log to both file and console for debugging

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
- **Try/Except Blocks**: Use specific exception types when possible
- **Graceful Degradation**: Continue operation when possible, log errors
- **API Failures**: Implement retry logic with exponential backoff
- **Resource Cleanup**: Use context managers for file/database connections

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
- **Dictionaries for Config**: Use nested dictionaries for configuration
- **DataFrames for OHLCV**: Use pandas DataFrames for market data
- **Named Tuples**: Consider namedtuples for complex return values

### Trading Logic Patterns
- **Position Tracking**: Use dictionary to track position state
- **Risk Management**: Always calculate position size based on risk parameters
- **Fee Calculation**: Account for trading fees in P/L calculations
- **Validation**: Validate inputs before executing trades

### Security Practices
- **API Keys**: Never commit API keys or secrets to repository
- **Paper Trading**: Always support paper trading mode for testing
- **Input Validation**: Validate all user inputs and API responses
- **Rate Limiting**: Respect exchange rate limits to avoid bans

### Testing Guidelines
- **Unit Tests**: Test individual functions with mock data
- **Integration Tests**: Test complete trading workflows
- **Mock Exchanges**: Use CCXT test exchanges for integration testing
- **Edge Cases**: Test with extreme market conditions

### Documentation
- **README Updates**: Update README.md for new features
- **Function Comments**: Document complex logic with inline comments
- **Configuration**: Document all configuration parameters

## Development Workflow

### Feature Development
1. Create feature branch from main
2. Implement changes following style guidelines
3. Run linting and formatting: `ruff check --fix . && ruff format .`
4. Test changes with paper trading mode
5. Run backtests to validate strategy
6. Update documentation
7. Commit with descriptive message

### Code Review Checklist
- [ ] Follows code style guidelines
- [ ] Includes appropriate error handling
- [ ] Uses logging for important operations
- [ ] Validates inputs and handles edge cases
- [ ] Includes docstrings for public functions
- [ ] Tested with paper trading mode
- [ ] Backtested for performance validation

## Project Structure
```
bfinance/
├── main.py                 # Main application entry point
├── ma_crossover_bot.py     # Moving average crossover strategy
├── backtest.py            # Backtesting framework
├── helper_functions.py    # Shared utility functions
├── requirements.txt       # Dependencies (legacy)
├── pyproject.toml         # Project configuration
├── uv.lock               # Lock file for uv
├── .env                  # Environment variables (not committed)
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## Performance Considerations
- **API Rate Limits**: Respect exchange rate limits (Binance: ~1200 req/min)
- **Memory Usage**: Be mindful of DataFrame sizes for long backtests
- **Network Efficiency**: Cache balance fetches and minimize API calls
- **Logging Overhead**: Use appropriate log levels to avoid performance impact

## Risk Management
- **Position Sizing**: Never risk more than specified percentage per trade
- **Stop Losses**: Always implement trailing or fixed stop losses
- **Paper Trading**: Test all changes in paper trading mode first
- **Backtesting**: Validate strategies with historical data before live trading

## Environment Setup
1. Clone repository: `git clone <repo-url>`
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install dependencies: `uv sync`
4. Create `.env` file with API credentials
5. Run in paper mode first: `PAPER_TRADING=true uv run ma_crossover_bot.py`</content>
<parameter name="filePath">/home/y/MY_PROJECTS/TradingBots/bfinance/AGENTS.md