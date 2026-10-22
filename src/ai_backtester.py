import pandas as pd

from src.ai_model import (
    build_model,
    get_features_for_market,
)


def backtest_ai_strategy(
    data,
    initial_capital=10000,
):
    """
    Walk-forward backtest of the Bojet AI strategy.

    Prediction:
        Using information available at day t's close,
        predict whether day t+1's close will exceed
        day t+1's open.

    Execution:
        UP   -> Buy at the next session's open and sell
                at that session's close.
        DOWN -> Remain in cash.

    The model is trained only on outcomes known by the
    prediction date.

    Limitations:
        - No transaction fees or slippage.
        - No taxes.
        - No interest earned on cash.
        - Assumes execution at the recorded opening price.
        - Uses a long-or-cash strategy, not short selling.

    Educational use only. Historical performance does not
    guarantee future results.
    """

    # --------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------

    if initial_capital <= 0:
        raise ValueError(
            "Initial capital must be greater than zero."
        )

    required_price_columns = ["Open", "Close"]

    missing_price_columns = [
        column
        for column in required_price_columns
        if column not in data.columns
    ]

    if missing_price_columns:
        raise ValueError(
            "Market data is missing required columns: "
            f"{missing_price_columns}"
        )

    data = data.copy().sort_index()

    # Avoid ambiguous results from duplicate timestamps.
    data = data.loc[
        ~data.index.duplicated(keep="last")
    ].copy()

    if len(data) < 60:
        raise ValueError(
            "At least 60 historical observations are required."
        )

    if data[["Open", "Close"]].isna().any().any():
        raise ValueError(
            "Open and Close prices contain missing values."
        )

    if (data[["Open", "Close"]] <= 0).any().any():
        raise ValueError(
            "Open and Close prices must be positive."
        )

    # --------------------------------------------------
    # CREATE NEXT-SESSION TARGET
    # --------------------------------------------------

    # These values describe the next actual trading session.
    data["Next_Open"] = data["Open"].shift(-1)
    data["Next_Close"] = data["Close"].shift(-1)

    data["Next_Date"] = pd.Series(
        data.index,
        index=data.index,
    ).shift(-1)

    # 1 = next session closes above its open.
    # 0 = next session closes at or below its open.
    data["Target"] = pd.Series(
        pd.NA,
        index=data.index,
        dtype="Int64",
    )

    valid_target = (
        data["Next_Open"].notna()
        & data["Next_Close"].notna()
    )

    data.loc[valid_target, "Target"] = (
        data.loc[valid_target, "Next_Close"]
        > data.loc[valid_target, "Next_Open"]
    ).astype(int)

    # --------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------

    features = get_features_for_market(data)

    missing_features = [
        feature
        for feature in features
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Required AI features are missing. "
            "Run calculate_indicators(data) first. "
            f"Missing features: {missing_features}"
        )

    required_columns = features + [
        "Next_Open",
        "Next_Close",
        "Next_Date",
        "Target",
    ]

    model_data = data.dropna(
        subset=required_columns
    ).copy()

    if len(model_data) < 60:
        raise ValueError(
            "Not enough complete observations after "
            "feature preparation."
        )

    if model_data["Target"].nunique() < 2:
        raise ValueError(
            "The available data contains only one target class."
        )

    # --------------------------------------------------
    # INITIALISE PORTFOLIO
    # --------------------------------------------------

    capital = float(initial_capital)

    portfolio_values = []
    execution_dates = []
    trades = []

    # Start after an initial historical training window.
    start_index = 50

    # --------------------------------------------------
    # WALK-FORWARD BACKTEST
    # --------------------------------------------------

    for i in range(start_index, len(model_data)):

        current_row = model_data.iloc[i]
        current_date = current_row.name

        # Only use labels whose outcomes were known by
        # the current prediction date.
        training_data = model_data.loc[
            model_data["Next_Date"] <= current_date
        ]

        if len(training_data) < 30:
            continue

        X_train = training_data[features]
        y_train = training_data["Target"].astype(int)

        # A classifier requires examples from both classes.
        if y_train.nunique() < 2:
            portfolio_values.append(capital)
            execution_dates.append(
                current_row["Next_Date"]
            )
            continue

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        # Today's features predict the next session.
        X_current = current_row[features].to_frame().T

        prediction = int(
            model.predict(X_current)[0]
        )

        # Execution occurs in the following session.
        execution_date = current_row["Next_Date"]

        open_price = float(
            current_row["Next_Open"]
        )

        close_price = float(
            current_row["Next_Close"]
        )

        if open_price <= 0 or close_price <= 0:
            raise ValueError(
                "Encountered a non-positive execution price."
            )

        # --------------------------------------------------
        # BUY AT OPEN, SELL AT CLOSE
        # --------------------------------------------------

        if prediction == 1:

            # Invest available capital at the opening price.
            capital_before_trade = capital
            shares = capital_before_trade / open_price

            # Value the position at the closing price.
            capital = shares * close_price

            profit = capital - capital_before_trade

            trades.append({
                "Date": execution_date,
                "Action": "BUY",
                "Price": open_price,
                "Shares": shares,
                "Profit": 0.0,
            })

            trades.append({
                "Date": execution_date,
                "Action": "SELL",
                "Price": close_price,
                "Shares": shares,
                "Profit": profit,
            })

        # If prediction is DOWN, remain in cash.
        # Capital does not change.

        portfolio_values.append(capital)
        execution_dates.append(execution_date)

    # --------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------

    if not portfolio_values:
        raise ValueError(
            "The backtest produced no portfolio results."
        )

    results = data.loc[execution_dates].copy()

    results["Portfolio"] = portfolio_values

    # Calculate daily portfolio returns.
    results["Daily_Return"] = (
        results["Portfolio"].pct_change()
    )

    # Compare the first result with initial capital.
    results.iloc[
        0,
        results.columns.get_loc("Daily_Return"),
    ] = (
        results["Portfolio"].iloc[0] / initial_capital
    ) - 1

    # --------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------

    final_value = float(
        results["Portfolio"].iloc[-1]
    )

    total_return = (
        (final_value - initial_capital)
        / initial_capital
    ) * 100

    # --------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------

    trades_df = pd.DataFrame(
        trades,
        columns=[
            "Date",
            "Action",
            "Price",
            "Shares",
            "Profit",
        ],
    )

    # Count only SELL records as completed trades.
    exit_trades = trades_df.loc[
        trades_df["Action"] == "SELL"
    ]

    winning_trades = int(
        (exit_trades["Profit"] > 0).sum()
    )

    losing_trades = int(
        (exit_trades["Profit"] < 0).sum()
    )

    completed_trades = len(exit_trades)

    if completed_trades:
        win_rate = (
            winning_trades / completed_trades
        ) * 100
    else:
        win_rate = 0.0

    # --------------------------------------------------
    # MAXIMUM DRAWDOWN
    # --------------------------------------------------

    # Include starting capital so an initial loss is counted.
    portfolio_curve = pd.Series(
        [initial_capital]
        + results["Portfolio"].tolist()
    )

    running_max = portfolio_curve.cummax()

    drawdown = (
        portfolio_curve - running_max
    ) / running_max

    max_drawdown = float(
        drawdown.min() * 100
    )

    # --------------------------------------------------
    # RETURN DASHBOARD-COMPATIBLE SUMMARY
    # --------------------------------------------------

    summary = {
        "initial_capital": float(initial_capital),
        "final_value": final_value,
        "total_return": float(total_return),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": float(win_rate),
        "max_drawdown": max_drawdown,
        "trades": trades_df,
    }

    return results, summary