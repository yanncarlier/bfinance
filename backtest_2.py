import pandas as pd
df = pd.read_csv('BTC-USD_daily.csv', parse_dates=['Date'])
df = df.set_index('Date')
initial_btc = 100 / df['Close'].iloc[0]
periods = [10, 20, 50, 100, 150, 200]
for n in periods:
    df[f'SMA{n}'] = df['Close'].rolling(n).mean()
    df[f'Position{n}'] = 0  # 1 = bought, 0 = cash
    df[f'Position{n}'].iloc[0:n] = 1  # Initial hold
    for t in range(n, len(df)):
        prev_close, curr_close = df['Close'].iloc[t-1], df['Close'].iloc[t]
        prev_sma, curr_sma = df[f'SMA{n}'].iloc[t-1], df[f'SMA{n}'].iloc[t]
        if df[f'Position{n}'].iloc[t-1] == 1 and prev_close >= prev_sma and curr_close < curr_sma:
            df[f'Position{n}'].iloc[t] = 0  # Sell
        elif df[f'Position{n}'].iloc[t-1] == 0 and prev_close <= prev_sma and curr_close > curr_sma:
            df[f'Position{n}'].iloc[t] = 1  # Buy
        else:
            df[f'Position{n}'].iloc[t] = df[f'Position{n}'].iloc[t-1]
    last_pos = df[f'Position{n}'].iloc[-1]
    if last_pos == 1:
        value = initial_btc * df['Close'].iloc[-1]
    else:
        value = 100  # Simplified; actual would track cash value
    print(f'SMA {n}: Final value ${value:.2f}')
