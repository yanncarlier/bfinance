"""
btc_price_tracker.py
====================
A simple, robust script to fetch and display the latest Bitcoin (BTC-USD) closing price
using Yahoo Finance via the `yfinance` library.
Features:
- Downloads 100 days of historical price data
- Extracts adjusted close prices
- Displays the last 10 closing prices
- Shows current price with clean formatting
- Calculates and displays 24-hour price change (in $ and %)
- Uses `.item()` for future-proof pandas scalar extraction
Author: You (with help from Grok)
Date: 2025-10-29
"""
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import yfinance as yf
"""
Import yfinance: A powerful, free library for downloading historical market data
from Yahoo Finance. Reliable for stocks, ETFs, and cryptocurrencies like BTC-USD.
"""
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
ticker = 'BTC-USD'
"""
Define the ticker symbol:
- 'BTC-USD' = Bitcoin priced in US Dollars
- Yahoo Finance uses this format for crypto pairs
- Case-sensitive and must match exactly
"""
# ──────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────
data = yf.download(
    ticker,
    period='100d',           # Fetch last 100 calendar days of data
    progress=False,          # Suppress download progress bar (clean output)
    auto_adjust=True         # Automatically adjust prices for splits/dividends
)
"""
Download historical market data from Yahoo Finance.
Parameters:
    ticker       → 'BTC-USD'
    period       → '100d' = 100 days (includes weekends; crypto is 24/7)
    progress     → False = no progress bar in console
    auto_adjust  → True  = 'Close' column is already adjusted (same as 'Adj Close')
Returns:
    pandas.DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    Index: DatetimeIndex (UTC timezone)
"""
# ──────────────────────────────────────────────────────────────────────────────
# EXTRACT CLOSE PRICES
# ──────────────────────────────────────────────────────────────────────────────
close_prices = data['Close']
"""
Extract the 'Close' column as a pandas Series.
- Since auto_adjust=True, 'Close' = adjusted close price
- For BTC-USD: No dividends/splits → 'Close' == 'Adj Close'
- Series index: Date (daily)
- Series values: Closing price at midnight UTC
"""
# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY LAST 10 DAYS
# ──────────────────────────────────────────────────────────────────────────────
print(close_prices.tail(10))
"""
Print the last 10 closing prices (most recent first).
- .tail(10) → last 10 rows of the Series
- Output format:
      Date
      2025-10-20    110588.93
      2025-10-21    108476.89
      ...
      Name: Close, dtype: float64
- Helpful for quick visual inspection of recent trend
"""
# ──────────────────────────────────────────────────────────────────────────────
# GET LATEST PRICE (FUTURE-PROOF)
# ──────────────────────────────────────────────────────────────────────────────
latest_price = close_prices.iloc[-1].item()
"""
Extract the most recent closing price as a native Python float.
Why this method?
    close_prices.iloc[-1]  → Returns a pandas scalar (with metadata)
    .item()               → Converts to pure Python float (recommended)
    → Avoids FutureWarning: "Calling float on a single-element Series is deprecated"
Benefits:
    - Future-proof (pandas ≥2.1)
    - Clean for formatting and math
    - No pandas overhead
"""
print(f"\nLatest BTC-USD Close: ${latest_price:,.2f}")
"""
Print the latest closing price with formatting:
    \n                    → New line for separation
    $                     → Dollar sign
    :,                    → Comma thousands separator
    .2f                   → 2 decimal places
Example output:
    Latest BTC-USD Close: $113,188.77
"""
# ──────────────────────────────────────────────────────────────────────────────
# CALCULATE 24-HOUR CHANGE
# ──────────────────────────────────────────────────────────────────────────────
prev_price = close_prices.iloc[-2].item()
"""
Get the previous day's closing price (24 hours ago).
- .iloc[-2] → second-to-last row
- .item() → convert to float
- For daily data, this represents ~24-hour change
"""
change_24h = latest_price - prev_price
"""
Calculate absolute price change over 24 hours.
- Positive = price went up
- Negative = price went down
"""
pct_change = (change_24h / prev_price) * 100
"""
Calculate percentage change over 24 hours.
Formula: ((new - old) / old) × 100
- +2.50% = up 2.5%
- -1.20% = down 1.2%
"""
# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY 24H CHANGE
# ──────────────────────────────────────────────────────────────────────────────
print(f"24h Change: ${change_24h:,.2f} ({pct_change:+.2f}%)")
"""
Print 24-hour change with clean formatting:
    ${change_24h:,.2f}     → Dollar amount with commas and 2 decimals
    ({pct_change:+.2f}%)  → Percentage with:
        + sign for positive
        2 decimal places
        % symbol
Example output:
    24h Change: $1,200.00 (+1.07%)
    24h Change: -$800.50 (-0.73%)
"""
# ──────────────────────────────────────────────────────────────────────────────
# END OF SCRIPT
# ──────────────────────────────────────────────────────────────────────────────
"""
This script is ideal for:
- Quick price checks
- Integration into dashboards
- Scheduled reports (via cron)
- Teaching pandas/yfinance basics
To run:
    uv run python btc_price_tracker.py
"""
