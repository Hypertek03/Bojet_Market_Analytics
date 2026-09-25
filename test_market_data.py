import pandas as pd

from market_data import get_market_data

# Get Apple stock data
data = get_market_data("AAPL")

# Moving Average
data["MA10"] = data["Close"].rolling(window=10).mean()

# RSI
delta = data["Close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

rs = average_gain / average_loss
data["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()

data["MACD"] = ema12 - ema26
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

# Bollinger Bands
data["BB_Middle"] = data["Close"].rolling(window=20).mean()
std = data["Close"].rolling(window=20).std()

data["BB_Upper"] = data["BB_Middle"] + (2 * std)
data["BB_Lower"] = data["BB_Middle"] - (2 * std)

# Remove rows with missing indicator values
data = data.dropna()

# Show the latest 5 days
print("\nBOJET MARKET ANALYSIS")
print("=" * 50)

latest = data[
    [
        "Close",
        "MA10",
        "RSI",
        "MACD",
        "Signal",
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
    ]
].tail(5)

print(latest.to_string())