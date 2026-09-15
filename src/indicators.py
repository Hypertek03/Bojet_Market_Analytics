def calculate_indicators(data):
    data = data.copy()

    # =========================
    # MOVING AVERAGES
    # =========================

    data["MA10"] = (
        data["Close"]
        .rolling(window=10)
        .mean()
    )

    data["MA20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["MA50"] = (
        data["Close"]
        .rolling(window=50)
        .mean()
    )

    # =========================
    # RSI
    # =========================

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    average_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    rs = average_gain / average_loss

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # =========================
    # MACD
    # =========================

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = (
        ema12 - ema26
    )

    data["Signal"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # =========================
    # BOLLINGER BANDS
    # =========================

    data["BB_Middle"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["BB_Std"] = (
        data["Close"]
        .rolling(window=20)
        .std()
    )

    data["BB_Upper"] = (
        data["BB_Middle"]
        + (2 * data["BB_Std"])
    )

    data["BB_Lower"] = (
        data["BB_Middle"]
        - (2 * data["BB_Std"])
    )

    # =========================
    # PRICE RATIOS
    # =========================

    data["Price_MA10_Ratio"] = (
        data["Close"]
        / data["MA10"]
    )

    data["Price_MA20_Ratio"] = (
        data["Close"]
        / data["MA20"]
    )

    data["Price_MA50_Ratio"] = (
        data["Close"]
        / data["MA50"]
    )

    data["Price_BB_Ratio"] = (
        data["Close"]
        / data["BB_Middle"]
    )

    # =========================
    # BOLLINGER POSITION
    # =========================

    data["BB_Position"] = (
        (data["Close"] - data["BB_Lower"])
        /
        (data["BB_Upper"] - data["BB_Lower"])
    )

    # =========================
    # MACD DIFFERENCE
    # =========================

    data["MACD_Difference"] = (
        data["MACD"]
        - data["Signal"]
    )

    # =========================
    # MOMENTUM
    # =========================

    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    data["Return_5D"] = (
        data["Close"].pct_change(5)
    )

    data["Return_10D"] = (
        data["Close"].pct_change(10)
    )

    # =========================
    # VOLATILITY
    # =========================

    data["Volatility_10D"] = (
        data["Daily_Return"]
        .rolling(window=10)
        .std()
    )

    # =========================
    # MOVING AVERAGE TREND
    # =========================

    data["MA10_MA20_Ratio"] = (
        data["MA10"]
        / data["MA20"]
    )

    data["MA20_MA50_Ratio"] = (
        data["MA20"]
        / data["MA50"]
    )

    # =========================
    # VOLUME
    # =========================

    if "Volume" in data.columns:

        data["Volume_Change"] = (
            data["Volume"]
            .pct_change()
        )

        data["Volume_MA20_Ratio"] = (
            data["Volume"]
            /
            data["Volume"]
            .rolling(window=20)
            .mean()
        )

    else:

        data["Volume_Change"] = 0

        data["Volume_MA20_Ratio"] = 1

    return data