import pandas as pd

from src.ai_model import (
    FEATURES,
    create_target,
)

from sklearn.ensemble import RandomForestClassifier


def backtest_ai_strategy(
    data,
    initial_capital=10000
):
    """
    Backtest the Bojet AI strategy.

    The AI uses today's information to predict
    whether the next trading day will move UP
    or DOWN.

    The trade is executed on the next trading day.
    """

    data = data.copy()

    data = create_target(data)

    data = data.dropna(
        subset=FEATURES + ["Next_Close"]
    ).copy()

    if len(data) < 60:
        raise ValueError(
            "Not enough historical data for AI backtesting."
        )

    capital = initial_capital
    shares = 0
    entry_price = None

    portfolio_values = []
    trades = []

    start_index = 50

    for i in range(start_index, len(data) - 1):

        # -----------------------------
        # TRAIN USING ONLY PAST DATA
        # -----------------------------

        training_data = data.iloc[:i].copy()

        X_train = training_data[FEATURES]
        y_train = training_data["Target"]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        # -----------------------------
        # AI PREDICTION
        # -----------------------------

        current_row = data.iloc[[i]]

        X_current = current_row[FEATURES]

        prediction = model.predict(
            X_current
        )[0]

        # -----------------------------
        # EXECUTE ON NEXT DAY
        # -----------------------------

        next_row = data.iloc[i + 1]

        price = next_row["Close"]
        date = next_row.name

        # -----------------------------
        # BUY
        # -----------------------------

        if prediction == 1 and shares == 0:

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

        # -----------------------------
        # SELL
        # -----------------------------

        elif prediction == 0 and shares > 0:

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

        # -----------------------------
        # PORTFOLIO VALUE
        # -----------------------------

        if shares > 0:

            portfolio_value = (
                shares * price
            )

        else:

            portfolio_value = capital

        portfolio_values.append(
            portfolio_value
        )

    # -----------------------------
    # FINAL EXIT
    # -----------------------------

    if shares > 0:

        final_price = data["Close"].iloc[-1]

        final_capital = (
            shares * final_price
        )

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

    # -----------------------------
    # RESULTS
    # -----------------------------

    results = data.iloc[
        start_index + 1:
    ].copy()

    results["Portfolio"] = (
        portfolio_values
    )

    results["Daily_Return"] = (
        results["Portfolio"].pct_change()
    )

    final_value = (
        results["Portfolio"].iloc[-1]
    )

    total_return = (
        (final_value - initial_capital)
        / initial_capital
    ) * 100

    # -----------------------------
    # TRADE STATISTICS
    # -----------------------------

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
        winning_trades
        + losing_trades
    )

    if completed_trades > 0:

        win_rate = (
            winning_trades
            / completed_trades
        ) * 100

    else:

        win_rate = 0

    # -----------------------------
    # MAX DRAWDOWN
    # -----------------------------

    running_max = (
        results["Portfolio"].cummax()
    )

    drawdown = (
        results["Portfolio"]
        - running_max
    ) / running_max

    max_drawdown = (
        drawdown.min() * 100
    )

    # -----------------------------
    # SUMMARY
    # -----------------------------

    summary = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": total_return,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "trades": trades_df
    }

    return results, summary