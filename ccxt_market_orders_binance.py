import ccxt
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
# === CONFIGURATION ===
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY and API_SECRET must be set in .env file")
# Initialize Binance exchange with rate limiting
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})
SYMBOL = 'BTC/USDT'
# === HELPER FUNCTIONS ===


def print_balance():
    """Fetch and print total BTC and USDT balance."""
    try:
        balance = exchange.fetch_balance()
        btc = balance['total'].get('BTC', 0)
        usdt = balance['total'].get('USDT', 0)
        print(f"Balance: {btc:.6f} BTC, {usdt:.2f} USDT")
        return btc, usdt
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None, None


def get_last_price(symbol):
    """Fetch the latest ticker price for a symbol."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker['last']
        print(f"Last price for {symbol}: {last_price}")
        return last_price
    except Exception as e:
        print(f"Error fetching ticker for {symbol}: {e}")
        return None


def place_market_order(side, amount):
    """
    Place a market order.
    side: 'buy' or 'sell'
    amount: amount in BTC
    """
    if side not in ['buy', 'sell']:
        print("Invalid side. Use 'buy' or 'sell'.")
        return None
    try:
        order = exchange.create_order(
            symbol=SYMBOL,
            type='market',
            side=side,
            amount=amount
        )
        print(f"Market {side.upper()} order placed successfully:")
        print(f"  Order ID: {order['id']}")
        print(f"  Amount: {amount} BTC")
        print(f"  Status: {order['status']}")
        return order
    except Exception as e:
        print(f"Error placing market {side} order: {e}")
        return None


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("=== Binance Trading Bot Demo ===\n")
    # 1. Print current balance
    print("1. Fetching account balance...")
    print_balance()
    print()
    # 2. Get current market price
    print("2. Fetching market data...")
    price = get_last_price(SYMBOL)
    print()
    # 3. Example: Place a small market buy order (uncomment to execute)
    """
    print("3. Placing market BUY order...")
    place_market_order(side='buy', amount=0.001)
    print()
    """
    # 4. Example: Place a small market sell order (uncomment to execute)
    """
    print("4. Placing market SELL order...")
    place_market_order(side='sell', amount=0.001)
    print()
    """
    print("Demo completed. Uncomment order lines to execute real trades.")
