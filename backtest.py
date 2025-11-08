import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from itertools import product
from concurrent.futures import ThreadPoolExecutor
# Load environment variables from .env file
load_dotenv()
# Retrieve Binance API key and secret
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
# Ensure credentials are available
if not API_KEY or not API_SECRET:
    raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
# =============================================================================
# BACKTEST CONFIGURATION (Backtest-Specific Variables)
# =============================================================================
BACKTEST_CONFIG = {
    'candle_timeframe': '5m',      # Candle timeframe
    'symbol': 'BTC/USDT',          # Trading pair
    'init_usdt': 10000.0,          # Initial USDT balance for backtesting
    'data_limit': 100000,          # Number of candles to fetch for backtesting
    'top_combos_to_display': 10,   # Number of top combinations to print
}
# =============================================================================
# STRATEGY CONFIGURATION (Shared with Live Trading; Override as Needed)
# =============================================================================
# These are backtest defaults but can be overridden (e.g., from LIVE_CONFIG if imported)
STRATEGY_CONFIG = {
    'position_size_pct': 0.02,     # Risk percentage of balance per trade
    'max_trade_usd': 1000.0,       # Maximum USD to risk per trade
    'stop_loss_pct': 0.05,         # Trailing stop-loss percentage
}
# =============================================================================
# CORE SETUP (Not specific to backtest)
# =============================================================================
# Initialize Binance exchange via CCXT
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
def fetch_ohlcv(symbol, timeframe, limit):
    """
    Fetch OHLCV data from Binance.
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Candle interval (e.g., '5m')
        limit (int): Number of candles to retrieve
    Returns:
        pd.DataFrame: OHLCV data indexed by timestamp, or None on failure
    """
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit)
        df = pd.DataFrame(
            raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        return None
def backtest_one(df, short_win, long_win,
                 init_usdt=BACKTEST_CONFIG['init_usdt'], pos_pct=STRATEGY_CONFIG['position_size_pct'],
                 max_usd=STRATEGY_CONFIG['max_trade_usd'], trail_pct=STRATEGY_CONFIG['stop_loss_pct']):
    """
    Backtest a single MA combination on historical data.
    Args:
        df (pd.DataFrame): OHLCV data with 'close' column
        short_win (int): Short MA window
        long_win (int): Long MA window
        init_usdt (float): Initial capital
        pos_pct (float): Position size percentage
        max_usd (float): Max USD per trade
        trail_pct (float): Trailing stop percentage
    Returns:
        dict: Performance metrics
    """
    close = df['close'].values
    n = len(close)
    short_ma = pd.Series(close).rolling(short_win).mean().values
    long_ma = pd.Series(close).rolling(long_win).mean().values
    cash = init_usdt
    btc = entry = peak = 0.0
    trades = []
    for i in range(max(short_win, long_win), n):
        price = close[i]
        if btc > 0:
            if price > peak:
                peak = price
            stop = peak * (1 - trail_pct)
            if price <= stop:
                cash += btc * price
                trades.append({'pnl': btc * (price - entry)})
                btc = entry = peak = 0.0
                continue
            if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                cash += btc * price
                trades.append({'pnl': btc * (price - entry)})
                btc = entry = peak = 0.0
                continue
        if btc == 0 and short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]:
            usd = min(cash * pos_pct, max_usd)
            btc = round(usd / price, 6)
            cash -= btc * price
            entry = peak = price
    if btc > 0:
        cash += btc * close[-1]
        trades.append({'pnl': btc * (close[-1] - entry)})
    pnl = cash - init_usdt
    ret = pnl / init_usdt * 100
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / len(trades) * 100 if trades else 0
    equity = np.cumsum([init_usdt] + [t['pnl'] for t in trades])
    dd = np.maximum.accumulate(equity) - equity
    max_dd = dd.max() / init_usdt * 100 if equity.size else 0
    return {
        'short': short_win, 'long': long_win,
        'return_%': round(ret, 3), 'trades': len(trades),
        'winrate_%': round(winrate, 1), 'max_dd_%': round(max_dd, 2),
        'final_usdt': round(cash, 2)
    }
def run_hf_backtest(
    limit=BACKTEST_CONFIG['data_limit'],
    symbol=BACKTEST_CONFIG['symbol'],
    timeframe=BACKTEST_CONFIG['candle_timeframe'],
    short_range=range(3, 16, 2),
    long_range=range(20, 81, 5),
    max_workers=12
):
    """
    Run grid search backtest over MA combinations.
    Args:
        limit (int): Number of candles to fetch
        symbol (str): Trading pair
        timeframe (str): Candle timeframe
        short_range (range): Short MA windows to test
        long_range (range): Long MA windows to test
        max_workers (int): Thread pool workers
    Returns:
        pd.DataFrame: Sorted results
    """
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is None or len(df) < 90:
        return pd.DataFrame()
    combos = list(product(short_range, long_range))
    results = []
    def worker(s, l):
        return backtest_one(df, s, l)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s, l in combos:
            res = ex.submit(worker, s, l).result()
            if res:
                results.append(res)
    out = pd.DataFrame(results).sort_values(
        'return_%', ascending=False).reset_index(drop=True)
    return out
def main():
    print("\nRunning backtest...")
    bt = run_hf_backtest()
    if not bt.empty:
        print(bt.head(BACKTEST_CONFIG['top_combos_to_display']).to_string(
            index=False))
        print(
            f"\nTop combo: {bt.iloc[0]['short']}/{bt.iloc[0]['long']} → {bt.iloc[0]['return_%']}%")
if __name__ == "__main__":
    main()
