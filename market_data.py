import yfinance as yf


def get_market_data(ticker_symbol="AAPL"):
    ticker = yf.Ticker(ticker_symbol)

    data = ticker.history(
        period="2y",
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("No market data was returned.")

    data = data.sort_index()

    return data