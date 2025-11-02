"""
BTC/USDT Daily Candlestick Chart Generator
-----------------------------------------
Fetches the last 90 days of BTC/USDT daily OHLCV data from Binance
using ccxt, converts it to a pandas DataFrame, prints a sample,
and saves a clean, professional candlestick chart as a PNG file.
Features:
- Rate limiting (respects Binance API rules)
- Proper timestamp handling (ISO 8601 to UTC)
- Clean output (no GUI/display issues)
- Warning suppression for dense data
- Saves chart to file (portable & reliable)
"""
# ===================================================================
# 1. IMPORTS
# ===================================================================
# Cryptocurrency exchange trading library (supports 100+ exchanges)
import ccxt
# Data manipulation and analysis (DataFrame handling)
import pandas as pd
# Matplotlib-based finance plotting (candles, volume, indicators)
import mplfinance as mpf
# For handling timestamps and UTC time
from datetime import datetime, timedelta, timezone
# ===================================================================
# 2. EXCHANGE INITIALIZATION
# ===================================================================
# Create a Binance exchange instance with rate limiting enabled
# 'enableRateLimit': True prevents hitting API rate limits by adding delays between requests
exchange = ccxt.binance({
    'enableRateLimit': True,  # Critical for public API usage — avoids HTTP 429 errors
})
# Load all available markets (symbols, precision, limits, etc.)
# This populates exchange.markets dictionary and enables symbol validation
exchange.load_markets()
# ===================================================================
# 3. CONFIGURATION PARAMETERS
# ===================================================================
# Trading pair: Bitcoin vs Tether (most liquid crypto pair)
symbol = 'BTC/USDT'
timeframe = '1d'              # Candlestick resolution: 1 day per candle
# start_date_str = '2022-09-01T00:00:00Z'  # Start date in ISO 8601 format (UTC)
# Calculate start date as 7 days ago from now
now = datetime.now(timezone.utc)
start_date = now - timedelta(days=7)
start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
# Start date in ISO 8601 format (UTC)
start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
# Convert ISO string to Unix timestamp in milliseconds (required by ccxt)
# parse8601() handles timezone-aware ISO strings correctly
start_date = int(exchange.parse8601(start_date_str))
# ===================================================================
# 4. USER FEEDBACK: Print fetch info
# ===================================================================
# Use exchange.iso8601() to convert timestamp back to clean YYYY-MM-DD
# [:10] extracts just the date part
print(
    f"Fetching {symbol} {timeframe} data from {exchange.iso8601(start_date)[:10]}...")
# ===================================================================
# 5. FETCH OHLCV DATA
# ===================================================================
# fetch_ohlcv() returns list of [timestamp, open, high, low, close, volume]
# limit=90: Fetch only the last 90 days (clean, readable chart)
# Note: Binance allows max 1000 candles per request
# raw_data = exchange.fetch_ohlcv(symbol, timeframe, limit=90)
raw_data = exchange.fetch_ohlcv(symbol, timeframe, limit=7)
# If no data is returned (e.g. invalid symbol), raise a clear error
if not raw_data:
    raise ValueError(
        f"No data returned for {symbol}. Check symbol and network connection.")
# ===================================================================
# 6. CONVERT TO PANDAS DATAFRAME
# ===================================================================
# Create DataFrame from raw list with meaningful column names
df = pd.DataFrame(raw_data, columns=[
    'timestamp',   # Milliseconds since Unix epoch (UTC)
    'open',        # Opening price of the period
    'high',        # Highest price during the period
    'low',         # Lowest price during the period
    'close',       # Closing price of the period
    'volume'       # Total traded volume (in base currency, i.e. BTC)
])
# Convert timestamp from milliseconds to datetime (UTC)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
# Set timestamp as index (required by mplfinance)
df.set_index('timestamp', inplace=True)
# Optional: Ensure correct column types (float for prices/volume)
df = df.astype({
    'open': 'float',
    'high': 'float',
    'low': 'float',
    'close': 'float',
    'volume': 'float'
})
# ===================================================================
# 7. DISPLAY SAMPLE DATA
# ===================================================================
# Show first 5 rows for verification
print("\nSample of fetched data (first 5 rows):")
print(df.head())
# ===================================================================
# 8. PLOT & SAVE CANDlestick CHART
# ===================================================================
# mpf.plot() configuration:
# - type='candle': Draw Japanese candlesticks
# - style='charles': Clean, high-contrast color scheme
# - title: Dynamic title with symbol and timeframe
# - volume=True: Include volume bars below price
# - savefig: Save chart to PNG file (works in headless/terminal/SSH)
# - warn_too_much_data=1000: Suppress "too much data" warning (we only have 90 points)
# mpf.plot(
#     df,
#     type='candle',                    # Candlestick style
#     style='charles',                  # Professional blue/red color theme
#     title=f'{symbol} {timeframe} (Last 90 Days)',
#     volume=True,                      # Show trading volume panel
#     # Output filename (saved in current directory)
#     savefig='btc_usdt_chart.png',
#     warn_too_much_data=1000,          # Avoid mplfinance density warning
#     # fignum=1                          # Reuse figure (optional, for scripting)
# )
# ===================================================================
# 9. FINAL CONFIRMATION
# # ===================================================================
# print("\nSuccess: Candlestick chart saved as 'btc_usdt_chart.png'")
# print("   Open the file in any image viewer or browser to see the chart.")
# ===================================================================
# END OF SCRIPT
# ===================================================================
