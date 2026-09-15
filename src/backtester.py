import pandas as pd


def backtest_strategy(
    data,
    initial_capital=10000
):
    """
    Backtest a simple MA10 trend-following strategy.

    Signal:
        BUY  when Close > MA10
        SELL when Close < MA10

    The signal is shifted by one day so that
    today's information is used for tomorrow's trade.
    """

    data = data.copy()

    if "MA10" not in data.columns:
        raise ValueError(
            "MA10 indicator is required for backtesting."
        )

    data = data.dropna(
        subset=["Close", "MA10"]
    ).copy()

    # Create trading signal
    data["Signal"] = 0

    data.loc[
        data["Close"] > data["MA10"],
        "Signal"
    ] = 1

    data.loc[
        data["Close"] < data["MA10"],
        "Signal"
    ] = 0

    # Use previous day's signal for today's position
    data["Position"] = data["Signal"].shift(1).fillna(0)

    capital = initial_capital
    shares = 0

    portfolio_values = []
    trades = []

    entry_price = None

    for date, row in data.iterrows():

        price = row["Close"]
        position = row["Position"]

        # BUY
        if position == 1 and shares == 0:

            shares = capital / price
            capital = 0
            entry_price = price

            trades.append({
                "Date": date,
                "Action": "BUY",
                "Price": price,
                "Shares": shares,
                "Profit": 0
            })

        # SELL
        elif position == 0 and shares > 0:

            capital = shares * price

            profit = (
                price - entry_price
            ) * shares

            trades.append({
                "Date": date,
                "Action": "SELL",
                "Price": price,
                "Shares": shares,
                "Profit": profit
            })

            shares = 0
            entry_price = None

        # Calculate portfolio value
        if shares > 0:
            portfolio_value = shares * price
        else:
            portfolio_value = capital

        portfolio_values.append(
            portfolio_value
        )

    # Close any open position
    if shares > 0:

        final_price = data["Close"].iloc[-1]

        final_capital = shares * final_price

        profit = (
            final_price - entry_price
        ) * shares

        trades.append({
            "Date": data.index[-1],
            "Action": "FINAL EXIT",
            "Price": final_price,
            "Shares": shares,
            "Profit": profit
        })

        capital = final_capital

        portfolio_values[-1] = final_capital

    # Store portfolio history
    results = data.copy()

    results["Portfolio"] = portfolio_values

    results["Daily_Return"] = (
        results["Portfolio"].pct_change()
    )

    # Performance
    final_value = results["Portfolio"].iloc[-1]

    total_return = (
        (final_value - initial_capital)
        / initial_capital
    ) * 100

    # Trade statistics
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:

        winning_trades = len(
            trades_df[
                trades_df["Profit"] > 0
            ]
        )

        losing_trades = len(
            trades_df[
                trades_df["Profit"] < 0
            ]
        )

    else:

        winning_trades = 0
        losing_trades = 0

    completed_trades = (
        winning_trades + losing_trades
    )

    if completed_trades > 0:

        win_rate = (
            winning_trades
            / completed_trades
        ) * 100

    else:

        win_rate = 0

    # Maximum drawdown
    running_max = (
        results["Portfolio"]
        .cummax()
    )

    drawdown = (
        results["Portfolio"]
        - running_max
    ) / running_max

    max_drawdown = (
        drawdown.min()
    ) * 100

    summary = {

        "initial_capital":
            initial_capital,

        "final_value":
            final_value,

        "total_return":
            total_return,

        "winning_trades":
            winning_trades,

        "losing_trades":
            losing_trades,

        "win_rate":
            win_rate,

        "max_drawdown":
            max_drawdown,

        "trades":
            trades_df
    }

    return results, summary

def backtest_buy_and_hold(
    data,
    initial_capital=10000
):
    """
    Backtest a simple Buy & Hold benchmark.

    Buys the asset at the beginning of the
    evaluation period and holds it until the end.
    """

    data = data.copy()

    data = data.dropna(
        subset=["Close"]
    ).copy()

    if data.empty:
        raise ValueError(
            "No price data available for Buy & Hold."
        )

    initial_price = data["Close"].iloc[0]

    shares = initial_capital / initial_price

    data["Portfolio"] = (
        data["Close"] * shares
    )

    data["Daily_Return"] = (
        data["Portfolio"].pct_change()
    )

    final_value = data["Portfolio"].iloc[-1]

    total_return = (
        (final_value - initial_capital)
        / initial_capital
    ) * 100

    running_max = data["Portfolio"].cummax()

    drawdown = (
        data["Portfolio"] - running_max
    ) / running_max

    max_drawdown = drawdown.min() * 100

    summary = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown
    }

    return data, summary