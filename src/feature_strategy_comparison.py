import pandas as pd

from src.ai_model import build_model, get_features_for_market
from src.indicators import calculate_indicators
from src.market_data import get_market_data


INITIAL_CAPITAL = 10_000
TRANSACTION_COST_RATE = 0.0005
START_INDEX = 50

CORE_FEATURES = [
    "Price_MA10_Ratio",
    "RSI",
    "MACD_Difference",
    "Price_BB_Ratio",
    "BB_Position",
]


def calculate_max_drawdown(values):
    values = pd.Series(values, dtype=float)

    running_max = values.cummax()
    drawdowns = (values - running_max) / running_max

    return float(drawdowns.min() * 100)


def run_feature_backtest(data, features, strategy_name):
    """Run a walk-forward intraday strategy for a feature group."""

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
            f"Not enough complete observations for {strategy_name}."
        )

    capital = float(INITIAL_CAPITAL)
    total_fees = 0.0
    portfolio_records = []
    trade_profits = []

    for i in range(START_INDEX, len(model_data)):
        current_row = model_data.iloc[i]
        current_date = current_row.name
        execution_date = current_row["Next_Date"]

        # Only use outcomes that were known by the prediction date.
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

        y_train = training_data["Target"].astype(int)

        if y_train.nunique() < 2:
            portfolio_records.append({
                "Date": execution_date,
                "Portfolio": capital,
                "Signal": "NO_TRADE",
            })
            continue

        model = build_model()
        model.fit(training_data[features], y_train)

        prediction = int(
            model.predict(current_row[features].to_frame().T)[0]
        )

        signal = "UP" if prediction == 1 else "DOWN"

        if prediction == 1:
            capital_before = capital

            buy_notional = capital_before / (
                1 + TRANSACTION_COST_RATE
            )
            buy_fee = buy_notional * TRANSACTION_COST_RATE
            shares = buy_notional / float(current_row["Next_Open"])

            sell_notional = (
                shares * float(current_row["Next_Close"])
            )
            sell_fee = sell_notional * TRANSACTION_COST_RATE

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

    if results.empty:
        raise ValueError(f"No results generated for {strategy_name}.")

    initial_and_daily_values = [
        INITIAL_CAPITAL,
        *results["Portfolio"].tolist(),
    ]

    final_value = float(results["Portfolio"].iloc[-1])
    completed_trades = len(trade_profits)

    summary = {
        "Strategy": strategy_name,
        "Final Value ($)": final_value,
        "Return (%)": (
            final_value / INITIAL_CAPITAL - 1
        ) * 100,
        "Total Fees ($)": total_fees,
        "Max Drawdown (%)": calculate_max_drawdown(
            initial_and_daily_values
        ),
        "Completed Trades": completed_trades,
        "Win Rate (%)": (
            sum(profit > 0 for profit in trade_profits)
            / completed_trades * 100
            if completed_trades
            else 0.0
        ),
    }

    return results, summary


def run_buy_and_hold(data):
    """Buy on the first evaluation open and hold to the final close."""

    first_open = float(data["Open"].iloc[0])
    final_close = float(data["Close"].iloc[-1])

    buy_notional = INITIAL_CAPITAL / (
        1 + TRANSACTION_COST_RATE
    )
    buy_fee = buy_notional * TRANSACTION_COST_RATE
    shares = buy_notional / first_open

    sell_notional = shares * final_close
    sell_fee = sell_notional * TRANSACTION_COST_RATE
    final_value = sell_notional - sell_fee

    daily_values = [
        INITIAL_CAPITAL,
        *(
            shares * data["Close"] * (
                1 - TRANSACTION_COST_RATE
            )
        ).tolist(),
    ]

    return {
        "Strategy": "Buy and Hold",
        "Final Value ($)": final_value,
        "Return (%)": (
            final_value / INITIAL_CAPITAL - 1
        ) * 100,
        "Total Fees ($)": buy_fee + sell_fee,
        "Max Drawdown (%)": calculate_max_drawdown(daily_values),
        "Completed Trades": 1,
        "Win Rate (%)": float("nan"),
    }


def run_always_up(data):
    """Buy every session open and sell at that session's close."""

    capital = float(INITIAL_CAPITAL)
    total_fees = 0.0
    portfolio_values = [capital]
    trade_profits = []

    for _, row in data.iterrows():
        capital_before = capital

        buy_notional = capital / (1 + TRANSACTION_COST_RATE)
        buy_fee = buy_notional * TRANSACTION_COST_RATE
        shares = buy_notional / float(row["Open"])

        sell_notional = shares * float(row["Close"])
        sell_fee = sell_notional * TRANSACTION_COST_RATE

        capital = sell_notional - sell_fee

        total_fees += buy_fee + sell_fee
        trade_profits.append(capital - capital_before)
        portfolio_values.append(capital)

    return {
        "Strategy": "Always UP",
        "Final Value ($)": capital,
        "Return (%)": (
            capital / INITIAL_CAPITAL - 1
        ) * 100,
        "Total Fees ($)": total_fees,
        "Max Drawdown (%)": calculate_max_drawdown(
            portfolio_values
        ),
        "Completed Trades": len(trade_profits),
        "Win Rate (%)": (
            sum(profit > 0 for profit in trade_profits)
            / len(trade_profits) * 100
            if trade_profits
            else 0.0
        ),
    }


def main():
    print("Downloading AAPL market data...")

    data = get_market_data(
        "AAPL",
        period="2y",
        interval="1d",
    )

    data = calculate_indicators(data)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")].copy()

    # Build labels and execution prices.
    data["Next_Open"] = data["Open"].shift(-1)
    data["Next_Close"] = data["Close"].shift(-1)
    data["Next_Date"] = data.index.to_series().shift(-1)

    data["Target"] = (
        data["Next_Close"] > data["Next_Open"]
    ).astype("Int64")

    invalid = (
        data["Next_Open"].isna()
        | data["Next_Close"].isna()
        | data["Next_Date"].isna()
    )

    data.loc[invalid, "Target"] = pd.NA

    available_features = get_features_for_market(data)

    if not set(CORE_FEATURES).issubset(available_features):
        raise ValueError(
            "One or more core features are unavailable in the data."
        )

    # Both AI models use exactly the same observations.
    common_data = data.dropna(
        subset=available_features + [
            "Next_Open",
            "Next_Close",
            "Next_Date",
            "Target",
        ]
    ).copy()

    common_data["Target"] = common_data["Target"].astype(int)

    feature_groups = [
        ("Core 5 features", CORE_FEATURES),
        ("All available features", available_features),
    ]

    ai_results = {}
    summaries = []

    for strategy_name, features in feature_groups:
        print(f"Running {strategy_name}...")

        results, summary = run_feature_backtest(
            common_data,
            features,
            strategy_name,
        )

        ai_results[strategy_name] = results
        summaries.append(summary)

    # Align benchmarks with the actual AI evaluation sessions.
    core_dates = ai_results["Core 5 features"].index
    all_dates = ai_results["All available features"].index

    if not core_dates.equals(all_dates):
        raise ValueError(
            "AI models do not share identical evaluation dates."
        )

    evaluation_data = data.reindex(core_dates).dropna(
        subset=["Open", "Close"]
    ).copy()

    if len(evaluation_data) != len(core_dates):
        raise ValueError(
            "Could not align all AI evaluation dates with price data."
        )

    summaries.append(run_buy_and_hold(evaluation_data))
    summaries.append(run_always_up(evaluation_data))

    comparison = pd.DataFrame(summaries)

    print("\nFEATURE STRATEGY COMPARISON")
    print("=" * 110)
    print(f"Starting capital: ${INITIAL_CAPITAL:,.2f}")
    print(
        "Transaction cost per side: "
        f"{TRANSACTION_COST_RATE * 100:.2f}%"
    )
    print(f"Evaluation sessions: {len(evaluation_data)}")
    print(f"First evaluation date: {evaluation_data.index[0]}")
    print(f"Last evaluation date: {evaluation_data.index[-1]}")
    print()
    print(comparison.round(2).to_string(index=False))

    print("\nImportant notes:")
    print("- These are historical simulations, not future guarantees.")
    print("- Results exclude spread, slippage, and taxes.")
    print(
        "- Buy and Hold, Always UP, and AI strategies have different "
        "market exposures."
    )


if __name__ == "__main__":
    main()
