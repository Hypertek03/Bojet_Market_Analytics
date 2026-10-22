import pandas as pd

from src.ai_model import build_model, get_features_for_market
from src.indicators import calculate_indicators
from src.market_data import get_market_data


INITIAL_CAPITAL = 10_000
TRANSACTION_COST_RATE = 0.0005  # 0.05% per transaction
TRAINING_SHARE = 0.80  # First 80% for historical development


def calculate_max_drawdown(values):
    """Calculate maximum drawdown, including starting capital."""
    values = pd.Series(values, dtype=float)

    running_max = values.cummax()
    drawdowns = (values - running_max) / running_max

    return float(drawdowns.min() * 100)


def prepare_model_data(data):
    """Prepare features and next-session outcomes."""
    data = data.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")].copy()

    data["Next_Open"] = data["Open"].shift(-1)
    data["Next_Close"] = data["Close"].shift(-1)
    data["Next_Date"] = data.index.to_series().shift(-1)

    data["Target"] = (
        data["Next_Close"] > data["Next_Open"]
    ).astype("Int64")

    invalid_target = (
        data["Next_Open"].isna()
        | data["Next_Close"].isna()
        | data["Next_Date"].isna()
    )

    data.loc[invalid_target, "Target"] = pd.NA

    features = get_features_for_market(data)

    model_data = data.dropna(
        subset=features + [
            "Next_Open",
            "Next_Close",
            "Next_Date",
            "Target",
        ]
    ).copy()

    model_data["Target"] = model_data["Target"].astype(int)

    if len(model_data) < 100:
        raise ValueError(
            "At least 100 complete observations are required "
            "for this holdout experiment."
        )

    return model_data, features


def run_holdout_backtest(model_data, features, initial_capital, cost_rate):
    """
    Test the final 20% using expanding-window training.

    Each prediction uses only outcomes known by the prediction date.
    """
    split_index = int(len(model_data) * TRAINING_SHARE)

    if split_index < 60 or len(model_data) - split_index < 20:
        raise ValueError("Insufficient observations for the holdout test.")

    capital = float(initial_capital)
    portfolio_records = []
    trade_profits = []
    total_fees = 0.0

    # Only execute trades in the final 20%.
    for i in range(split_index, len(model_data)):
        current_row = model_data.iloc[i]
        current_date = current_row.name
        execution_date = current_row["Next_Date"]

        # Only use outcomes that were known by current_date.
        training_data = model_data.loc[
            model_data["Next_Date"] <= current_date
        ]

        if len(training_data) < 30:
            raise ValueError(
                "Insufficient historical training observations."
            )

        X_train = training_data[features]
        y_train = training_data["Target"]

        if y_train.nunique() < 2:
            raise ValueError(
                "Training data contains only one target class."
            )

        model = build_model()
        model.fit(X_train, y_train)

        X_current = current_row[features].to_frame().T
        prediction = int(model.predict(X_current)[0])

        signal = "UP" if prediction == 1 else "DOWN"

        if prediction == 1:
            capital_before = capital

            # Account for the purchase fee before sizing the position.
            buy_notional = capital_before / (1 + cost_rate)
            buy_fee = buy_notional * cost_rate

            open_price = float(current_row["Next_Open"])
            close_price = float(current_row["Next_Close"])

            shares = buy_notional / open_price

            sell_notional = shares * close_price
            sell_fee = sell_notional * cost_rate

            capital = sell_notional - sell_fee

            total_fees += buy_fee + sell_fee
            trade_profits.append(capital - capital_before)

        portfolio_records.append({
            "Date": execution_date,
            "Portfolio": capital,
            "Signal": signal,
        })

    results = pd.DataFrame(portfolio_records).set_index("Date")
    results = results.sort_index()

    final_value = float(results["Portfolio"].iloc[-1])

    total_return = (
        (final_value / initial_capital) - 1
    ) * 100

    portfolio_values = [
        float(initial_capital),
        *results["Portfolio"].tolist(),
    ]

    winning_trades = sum(profit > 0 for profit in trade_profits)
    completed_trades = len(trade_profits)

    summary = {
        "Strategy": "Bojet AI",
        "Initial Capital": initial_capital,
        "Final Value": final_value,
        "Return (%)": total_return,
        "Total Fees": total_fees,
        "Max Drawdown (%)": calculate_max_drawdown(
            portfolio_values
        ),
        "Completed Trades": completed_trades,
        "Win Rate (%)": (
            winning_trades / completed_trades * 100
            if completed_trades
            else 0.0
        ),
    }

    return results, summary


def backtest_buy_and_hold(data, initial_capital, cost_rate):
    """Buy at the first open and hold until the final close."""
    first_open = float(data["Open"].iloc[0])
    final_close = float(data["Close"].iloc[-1])

    buy_notional = initial_capital / (1 + cost_rate)
    buy_fee = buy_notional * cost_rate

    shares = buy_notional / first_open

    final_value = (
        shares * final_close * (1 - cost_rate)
    )

    daily_values = (
        shares * data["Close"] * (1 - cost_rate)
    ).tolist()

    return {
        "Strategy": "Buy and Hold",
        "Initial Capital": initial_capital,
        "Final Value": final_value,
        "Return (%)": (
            (final_value / initial_capital) - 1
        ) * 100,
        "Total Fees": buy_fee + shares * final_close * cost_rate,
        "Max Drawdown (%)": calculate_max_drawdown(
            [initial_capital] + daily_values
        ),
        "Completed Trades": 1,
        "Win Rate (%)": None,
    }


def backtest_always_up(data, initial_capital, cost_rate):
    """Buy each session's open and sell at that session's close."""
    capital = float(initial_capital)
    portfolio_values = [capital]
    total_fees = 0.0
    completed_trades = 0
    winning_trades = 0

    for _, row in data.iterrows():
        capital_before = capital

        buy_notional = capital / (1 + cost_rate)
        buy_fee = buy_notional * cost_rate

        shares = buy_notional / float(row["Open"])

        sell_notional = shares * float(row["Close"])
        sell_fee = sell_notional * cost_rate

        capital = sell_notional - sell_fee

        total_fees += buy_fee + sell_fee
        completed_trades += 1

        if capital > capital_before:
            winning_trades += 1

        portfolio_values.append(capital)

    return {
        "Strategy": "Always UP",
        "Initial Capital": initial_capital,
        "Final Value": capital,
        "Return (%)": (
            (capital / initial_capital) - 1
        ) * 100,
        "Total Fees": total_fees,
        "Max Drawdown (%)": calculate_max_drawdown(
            portfolio_values
        ),
        "Completed Trades": completed_trades,
        "Win Rate (%)": (
            winning_trades / completed_trades * 100
            if completed_trades
            else 0.0
        ),
    }


def main():
    print("Downloading AAPL market data...")

    data = get_market_data("AAPL", period="2y", interval="1d")
    data = calculate_indicators(data)

    model_data, features = prepare_model_data(data)

    split_index = int(len(model_data) * TRAINING_SHARE)
    holdout_start = model_data["Next_Date"].iloc[split_index]
    holdout_end = model_data["Next_Date"].iloc[-1]

    print(f"Usable observations: {len(model_data)}")
    print(f"Features: {len(features)}")
    print(f"Initial development share: {TRAINING_SHARE:.0%}")
    print(f"Holdout start: {holdout_start}")
    print(f"Holdout end: {holdout_end}")
    print(f"Transaction cost per side: {TRANSACTION_COST_RATE:.2%}")
    print("\nRunning holdout test...")

    ai_results, ai_summary = run_holdout_backtest(
        model_data,
        features,
        INITIAL_CAPITAL,
        TRANSACTION_COST_RATE,
    )

    # Align both benchmarks to the AI's exact execution dates.
    evaluation_data = data.reindex(ai_results.index)

    if evaluation_data[["Open", "Close"]].isna().any().any():
        raise ValueError(
            "Missing prices on one or more AI evaluation dates."
        )

    buy_hold = backtest_buy_and_hold(
        evaluation_data,
        INITIAL_CAPITAL,
        TRANSACTION_COST_RATE,
    )

    always_up = backtest_always_up(
        evaluation_data,
        INITIAL_CAPITAL,
        TRANSACTION_COST_RATE,
    )

    results = pd.DataFrame([
        ai_summary,
        buy_hold,
        always_up,
    ])

    print("\nHOLDOUT TEST RESULTS")
    print("=" * 100)
    print(results.round(2).to_string(index=False))

    print("\nThis test evaluates the final 20% of usable observations.")
    print(
        "The model trains on past outcomes as they become available. "
        "The holdout period is not used to choose features or tune settings."
    )
    print(
        "These are historical simulations, not guarantees of future results."
    )


if __name__ == "__main__":
    main()