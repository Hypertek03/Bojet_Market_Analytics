import pandas as pd

from src.ai_backtester_costs import backtest_ai_strategy_with_costs
from src.indicators import calculate_indicators
from src.market_data import get_market_data


INITIAL_CAPITAL = 10_000
TRANSACTION_COST_RATE = 0.0005  # 0.05% per transaction


def calculate_max_drawdown(values):
    """Calculate maximum drawdown, including initial capital."""
    values = pd.Series(values, dtype=float)

    running_max = values.cummax()
    drawdowns = (values - running_max) / running_max

    return float(drawdowns.min() * 100)


def backtest_buy_and_hold(data, initial_capital, cost_rate):
    """Buy on the first evaluation day's open and hold to the end."""
    first_open = float(data["Open"].iloc[0])
    final_close = float(data["Close"].iloc[-1])

    buy_notional = initial_capital / (1 + cost_rate)
    buy_fee = buy_notional * cost_rate

    shares = buy_notional / first_open

    sell_notional = shares * final_close
    sell_fee = sell_notional * cost_rate

    final_value = sell_notional - sell_fee

    # Estimate daily liquidation value after the selling fee.
    daily_values = (
        shares * data["Close"] * (1 - cost_rate)
    ).tolist()

    max_drawdown = calculate_max_drawdown(
        [initial_capital] + daily_values
    )

    return {
        "Strategy": "Buy and Hold",
        "Initial Capital": initial_capital,
        "Final Value": final_value,
        "Return (%)": (
            (final_value / initial_capital) - 1
        ) * 100,
        "Total Fees": buy_fee + sell_fee,
        "Max Drawdown (%)": max_drawdown,
        "Completed Trades": 1,
    }


def backtest_always_up(data, initial_capital, cost_rate):
    """Buy every evaluation day's open and sell at its close."""
    capital = float(initial_capital)
    total_fees = 0.0
    portfolio_values = [capital]
    completed_trades = 0

    for _, row in data.iterrows():
        open_price = float(row["Open"])
        close_price = float(row["Close"])

        buy_notional = capital / (1 + cost_rate)
        buy_fee = buy_notional * cost_rate

        shares = buy_notional / open_price

        sell_notional = shares * close_price
        sell_fee = sell_notional * cost_rate

        capital = sell_notional - sell_fee

        total_fees += buy_fee + sell_fee
        completed_trades += 1
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
    }


def main():
    print("Downloading AAPL market data...")

    data = get_market_data("AAPL", period="2y", interval="1d")
    data = calculate_indicators(data)
    data = data.sort_index()

    print("Running Bojet AI backtest...")

    ai_results, ai_summary = backtest_ai_strategy_with_costs(
        data,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )

    # Align both benchmarks to the AI's actual execution dates.
    evaluation_data = data.reindex(ai_results.index)

    evaluation_data = evaluation_data.dropna(
        subset=["Open", "Close"]
    ).copy()

    if evaluation_data.empty:
        raise ValueError(
            "No matching market data was found for the AI evaluation dates."
        )

    if len(evaluation_data) != len(ai_results):
        raise ValueError(
            "The benchmark dates do not fully match the AI evaluation dates."
        )

    print(f"Matched evaluation sessions: {len(evaluation_data)}")
    print(f"First evaluation date: {evaluation_data.index[0]}")
    print(f"Last evaluation date: {evaluation_data.index[-1]}")
    print(f"Starting capital: ${INITIAL_CAPITAL:,.2f}")
    print(
        "Transaction cost per side: "
        f"{TRANSACTION_COST_RATE * 100:.2f}%"
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

    ai_result = {
        "Strategy": "Bojet AI",
        "Initial Capital": ai_summary["initial_capital"],
        "Final Value": ai_summary["final_value"],
        "Return (%)": ai_summary["total_return"],
        "Total Fees": ai_summary["total_fees"],
        "Max Drawdown (%)": ai_summary["max_drawdown"],
        "Completed Trades": ai_summary["completed_trades"],
    }

    results = pd.DataFrame(
        [ai_result, buy_hold, always_up]
    )

    print("\nBOJET AI VS BENCHMARKS")
    print("=" * 100)
    print(results.round(2).to_string(index=False))

    print("\nImportant:")
    print(
        "All three strategies use the same evaluation dates and "
        "starting capital."
    )
    print(
        "Results are historical simulations, not guarantees of "
        "future performance."
    )


if __name__ == "__main__":
    main()