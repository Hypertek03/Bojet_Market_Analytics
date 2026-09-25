import yfinance as yf


def get_market_data(
    symbol,
    period="1y",
    interval="1d"
):
    """
    Download historical market data.

    Works with stocks, forex and crypto
    symbols supported by Yahoo Finance.
    """

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No market data found for {symbol}."
        )

    # Handle Yahoo Finance MultiIndex columns
    if hasattr(data.columns, "nlevels"):
        if data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

    data = data.dropna().copy()

    return data