import pandas as pd

from src.ai_model import build_model, get_features_for_market


def backtest_ai_strategy_with_costs(
    data,
    initial_capital=10_000,
    transaction_cost_rate=0.0005,
):
    """
    Run a walk-forward AI backtest with transaction costs.

    Strategy:
        1. Train using only outcomes known at the prediction date.
        2. Predict whether the next session's close will exceed its open.
        3. If UP, buy at the next open and sell at the next close.
        4. If DOWN, remain in cash.

    Transaction costs are charged on both the purchase and sale.

    Returns:
        results: DataFrame containing portfolio history.
        summary: Dictionary of performance statistics and trade records.
    """

    if initial_capital <= 0:
        raise ValueError("Initial capital must be greater than zero.")

    if not 0 <= transaction_cost_rate < 1:
        raise ValueError(
            "Transaction cost rate must be at least 0 and less than 1."
        )

    data = data.copy().sort_index()

    required_columns = {"Open", "Close"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required price columns: {sorted(missing_columns)}"
        )

    data = data[~data.index.duplicated(keep="last")].copy()
    data = data.dropna(subset=["Open", "Close"]).copy()

    if len(data) < 60:
        raise ValueError("At least 60 price observations are required.")

    if (data["Open"] <= 0).any() or (data["Close"] <= 0).any():
        raise ValueError("Open and Close prices must be positive.")

    # Build the next-session target and execution prices.
    data["Next_Open"] = data["Open"].shift(-1)
    data["Next_Close"] = data["Close"].shift(-1)
    data["Next_Date"] = data.index.to_series().shift(-1)

    data["Target"] = (
        data["Next_Close"] > data["Next_Open"]
    ).astype("Int64")

    # The last row has no next session to trade.
    invalid_target = (
        data["Next_Open"].isna()
        | data["Next_Close"].isna()
        | data["Next_Date"].isna()
    )
    data.loc[invalid_target, "Target"] = pd.NA

    # Select the same available features as the existing AI model.
    features = get_features_for_market(data)

    missing_features = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required indicator features: {missing_features}"
        )

    model_data = data.dropna(
        subset=features + [
            "Next_Open",
            "Next_Close",
            "Next_Date",
            "Target",
        ]
    ).copy()

    model_data["Target"] = model_data["Target"].astype(int)

    if len(model_data) < 60:
        raise ValueError(
            "At least 60 complete observations are required."
        )

    if model_data["Target"].nunique() < 2:
        raise ValueError("The target must contain both UP and DOWN outcomes.")

    capital = float(initial_capital)
    start_index = 50

    portfolio_records = []
    trade_records = []
    trade_profits = []
    total_fees = 0.0

    for i in range(start_index, len(model_data)):
        current_row = model_data.iloc[i]
        current_date = current_row.name
        execution_date = current_row["Next_Date"]

        # Train only on next-session outcomes known by current_date.
        training_data = model_data.loc[
            model_data["Next_Date"] <= current_date
        ]

        if len(training_data) < 30:
            portfolio_records.append({
                "Date": execution_date,
                "Portfolio": capital,
                "Signal": "NO_TRADE",
            })
            continue

        X_train = training_data[features]
        y_train = training_data["Target"].astype(int)

        if y_train.nunique() < 2:
            portfolio_records.append({
                "Date": execution_date,
                "Portfolio": capital,
                "Signal": "NO_TRADE",
            })
            continue

        model = build_model()
        model.fit(X_train, y_train)

        X_current = current_row[features].to_frame().T
        prediction = int(model.predict(X_current)[0])

        open_price = float(current_row["Next_Open"])
        close_price = float(current_row["Next_Close"])

        signal = "UP" if prediction == 1 else "DOWN"

        if prediction == 1:
            capital_before = capital

            # Spend the available capital, including the purchase fee.
            buy_notional = capital_before / (
                1 + transaction_cost_rate
            )
            buy_fee = buy_notional * transaction_cost_rate
            shares = buy_notional / open_price

            # Sell all shares at the session close.
            sell_notional = shares * close_price
            sell_fee = sell_notional * transaction_cost_rate

            capital = sell_notional - sell_fee

            total_fees += buy_fee + sell_fee
            net_profit = capital - capital_before
            trade_profits.append(net_profit)

            trade_records.append({
                "Date": execution_date,
                "Action": "BUY",
                "Price": open_price,
                "Shares": shares,
                "Notional": buy_notional,
                "Fee": buy_fee,
            })

            trade_records.append({
                "Date": execution_date,
                "Action": "SELL",
                "Price": close_price,
                "Shares": shares,
                "Notional": sell_notional,
                "Fee": sell_fee,
                "Net_Profit": net_profit,
            })

        # For DOWN predictions, capital stays in cash.
        portfolio_records.append({
            "Date": execution_date,
            "Portfolio": capital,
            "Signal": signal,
        })

    if not portfolio_records:
        raise ValueError("No portfolio observations were generated.")

    results = pd.DataFrame(portfolio_records)
    results = results.set_index("Date").sort_index()

    results["Daily_Return"] = results["Portfolio"].pct_change().fillna(0)

    final_value = float(results["Portfolio"].iloc[-1])

    total_return = (
        (final_value - initial_capital) / initial_capital
    ) * 100

    # Include starting capital in the drawdown calculation.
    portfolio_values = pd.Series(
        [float(initial_capital)] + results["Portfolio"].tolist(),
        dtype=float,
    )

    running_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = float(drawdowns.min() * 100)

    completed_trades = len(trade_profits)
    winning_trades = sum(profit > 0 for profit in trade_profits)
    losing_trades = sum(profit <= 0 for profit in trade_profits)

    win_rate = (
        winning_trades / completed_trades * 100
        if completed_trades > 0
        else 0.0
    )

    summary = {
        "initial_capital": float(initial_capital),
        "final_value": final_value,
        "total_return": total_return,
        "total_fees": total_fees,
        "completed_trades": completed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "transaction_cost_rate": transaction_cost_rate,
        "trades": pd.DataFrame(trade_records),
    }

    return results, summary
