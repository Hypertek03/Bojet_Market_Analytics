from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.ai_model import build_model, get_features_for_market
from src.indicators import calculate_indicators
from src.market_data import get_market_data


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "AAPL"
PERIOD = "2y"
INTERVAL = "1d"

INITIAL_TRAINING_SAMPLES = 60

# New York time is used because AAPL trades on the US market.
MARKET_TIMEZONE = ZoneInfo("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "forward_predictions.csv"


# ============================================================
# MARKET DATA
# ============================================================

def get_completed_market_data(data):
    """
    Return daily candles considered complete.

    Before 4:30 PM New York time, exclude today's candle.
    After 4:30 PM, allow today's candle if Yahoo Finance
    has returned it, providing a buffer after market close.
    """

    data = data.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")].copy()

    now_ny = datetime.now(MARKET_TIMEZONE)
    today_ny = now_ny.date()

    session_dates = pd.Index(
        pd.to_datetime(data.index).date
    )

    # Allow today's candle only after the market has closed
    # and a 30-minute buffer has passed.
    after_close_buffer = (
        now_ny.hour > 16
        or (now_ny.hour == 16 and now_ny.minute >= 30)
    )

    if after_close_buffer:
        data = data.loc[session_dates <= today_ny].copy()
    else:
        data = data.loc[session_dates < today_ny].copy()

    if data.empty:
        raise ValueError(
            "No completed daily candles are available. "
            "Try again later."
        )

    return data


# ============================================================
# PREDICTION LOG
# ============================================================

def load_prediction_log():
    """
    Load the persistent CSV log.

    Existing predictions are retained. Missing optional columns
    are added to accommodate an older version of the log.
    """

    columns = [
        "Symbol",
        "Signal_Date",
        "Outcome_Date",
        "Signal",
        "Up_Probability",
        "Down_Probability",
        "Confidence",
        "Actual",
        "Correct",
    ]

    if not LOG_FILE.exists():
        return pd.DataFrame(columns=columns)

    log = pd.read_csv(LOG_FILE)

    for column in columns:
        if column not in log.columns:
            log[column] = np.nan

        log = log[columns].copy()

    for column in ["Signal_Date", "Outcome_Date"]:
        log[column] = (
            log[column]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    return log


def save_prediction_log(log):
    """Save predictions and outcomes to CSV."""

    log.to_csv(LOG_FILE, index=False)


# ============================================================
# TRAINING DATA
# ============================================================

def prepare_labeled_data(data, features):
    """
    Create the target for each session.

    Target = 1 when the NEXT session closes above its open.
    Target = 0 otherwise.

    The last row has no known next-session outcome and is not
    eligible for historical training.
    """

    labeled = data.copy()

    labeled["Next_Open"] = labeled["Open"].shift(-1)
    labeled["Next_Close"] = labeled["Close"].shift(-1)
    labeled["Next_Date"] = pd.Series(
        labeled.index,
        index=labeled.index,
    ).shift(-1)

    valid = (
        labeled["Next_Open"].notna()
        & labeled["Next_Close"].notna()
        & labeled["Next_Date"].notna()
    )

    labeled["Target"] = np.nan

    labeled.loc[valid, "Target"] = (
        labeled.loc[valid, "Next_Close"]
        > labeled.loc[valid, "Next_Open"]
    ).astype(int)

    labeled = labeled.dropna(
        subset=features + ["Target", "Next_Date"]
    ).copy()

    labeled["Target"] = labeled["Target"].astype(int)

    return labeled


# ============================================================
# SETTLE PENDING PREDICTIONS
# ============================================================

def update_previous_outcomes(log, completed_data):
    """
    For each pending prediction, find the first market session
    after its signal date.

    Only sessions in completed_data can settle a prediction.
    Existing settled results are never overwritten.
    """

    if log.empty:
        return log

    market_dates = pd.DatetimeIndex(completed_data.index)

    for row_index, record in log.iterrows():

        # Only process predictions for this symbol.
        if record["Symbol"] != SYMBOL:
            continue

        # Do not modify a prediction that already has an outcome.
        actual_value = record["Actual"]

        if pd.notna(actual_value) and str(actual_value).strip() != "":
            continue

        signal_date = pd.Timestamp(record["Signal_Date"]).normalize()

        # Locate the next available session strictly after signal date.
        future_dates = market_dates[market_dates.normalize() > signal_date]

        if len(future_dates) == 0:
            continue

        outcome_timestamp = future_dates[0]
        outcome_row = completed_data.loc[outcome_timestamp]

        open_price = float(outcome_row["Open"])
        close_price = float(outcome_row["Close"])

        if (
            not np.isfinite(open_price)
            or not np.isfinite(close_price)
            or open_price <= 0
        ):
            continue

        actual = int(close_price > open_price)

        signal = str(record["Signal"]).strip().upper()

        if signal not in {"UP", "DOWN"}:
            print(
                f"Skipping invalid signal for "
                f"{record['Signal_Date']}: {signal}"
            )
            continue

        predicted = 1 if signal == "UP" else 0

        log.at[row_index, "Outcome_Date"] = (
            pd.Timestamp(outcome_timestamp).strftime("%Y-%m-%d")
        )
        log.at[row_index, "Actual"] = actual
        log.at[row_index, "Correct"] = int(predicted == actual)

        actual_label = "UP" if actual == 1 else "DOWN"
        result_label = "CORRECT" if predicted == actual else "INCORRECT"

        print("\nPREDICTION SETTLED")
        print("-" * 50)
        print(f"Signal date: {record['Signal_Date']}")
        print(f"Outcome date: {outcome_timestamp:%Y-%m-%d}")
        print(f"Predicted: {signal}")
        print(f"Actual: {actual_label}")
        print(f"Result: {result_label}")
        print(f"Open: ${open_price:.2f}")
        print(f"Close: ${close_price:.2f}")

    return log


# ============================================================
# GENERATE A NEW PREDICTION
# ============================================================

def generate_prediction(data, features, log):
    """Train the model and record one prediction per signal date."""

    signal_timestamp = data.index[-1]
    signal_date = signal_timestamp.strftime("%Y-%m-%d")

    existing = log[
        (log["Symbol"] == SYMBOL)
        & (log["Signal_Date"] == signal_date)
    ]

    if not existing.empty:
        print(
            f"\nA prediction for {SYMBOL} on {signal_date} "
            "already exists. No duplicate was created."
        )
        return log

    labeled = prepare_labeled_data(data, features)

    # Only labels whose outcomes are known by the signal date
    # may be used for training.
    signal_date_normalized = pd.Timestamp(signal_timestamp).normalize()

    training_data = labeled[
        pd.to_datetime(labeled["Next_Date"]).dt.normalize()
        <= signal_date_normalized
    ].copy()

    if len(training_data) < INITIAL_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {len(training_data)} training observations are "
            f"available; at least {INITIAL_TRAINING_SAMPLES} are required."
        )

    if training_data["Target"].nunique() < 2:
        raise ValueError(
            "Training data contains fewer than two target classes."
        )

    latest_features = data.iloc[[-1]][features]

    model = build_model()

    model.fit(
        training_data[features],
        training_data["Target"],
    )

    probabilities = model.predict_proba(latest_features)[0]
    classes = list(model.classes_)

    up_probability = (
        float(probabilities[classes.index(1)])
        if 1 in classes
        else 0.0
    )

    down_probability = (
        float(probabilities[classes.index(0)])
        if 0 in classes
        else 0.0
    )

    predicted_class = int(model.predict(latest_features)[0])
    signal = "UP" if predicted_class == 1 else "DOWN"

    confidence = max(up_probability, down_probability)

    new_record = {
        "Symbol": SYMBOL,
        "Signal_Date": signal_date,
        "Outcome_Date": "",
        "Signal": signal,
        "Up_Probability": round(up_probability * 100, 2),
        "Down_Probability": round(down_probability * 100, 2),
        "Confidence": round(confidence * 100, 2),
        "Actual": np.nan,
        "Correct": np.nan,
    }

    log = pd.concat(
        [log, pd.DataFrame([new_record])],
        ignore_index=True,
    )

    print("\nNEW PAPER PREDICTION")
    print("=" * 55)
    print(f"Symbol: {SYMBOL}")
    print(f"Signal date: {signal_date}")
    print(f"Training observations: {len(training_data)}")
    print(f"Features: {len(features)}")
    print(f"Signal: {signal}")
    print(f"UP probability: {up_probability * 100:.2f}%")
    print(f"DOWN probability: {down_probability * 100:.2f}%")
    print(f"Model confidence: {confidence * 100:.2f}%")

    return log


# ============================================================
# EVALUATION
# ============================================================

def print_evaluation(log):
    """Evaluate only predictions with known, completed outcomes."""

    if log.empty:
        print("\nNo predictions have been recorded yet.")
        return

    actual_numeric = pd.to_numeric(log["Actual"], errors="coerce")

    completed = log[actual_numeric.notna()].copy()
    pending = len(log) - len(completed)

    print("\nFORWARD-TEST SUMMARY")
    print("=" * 55)
    print(f"Total recorded predictions: {len(log)}")
    print(f"Completed outcomes: {len(completed)}")
    print(f"Pending outcomes: {pending}")

    if completed.empty:
        print(
            "\nNo completed outcomes yet. Run this script again "
            "when the next market session is available as a "
            "completed daily candle."
        )
        return

    actual = pd.to_numeric(completed["Actual"]).astype(int).to_numpy()

    predicted = (
        completed["Signal"]
        .map({"DOWN": 0, "UP": 1})
        .astype(int)
        .to_numpy()
    )

    up_probabilities = (
        pd.to_numeric(completed["Up_Probability"]).to_numpy() / 100
    )

    print(f"Accuracy: {accuracy_score(actual, predicted) * 100:.2f}%")
    print(f"Brier score: {brier_score_loss(actual, up_probabilities):.4f}")

    if len(np.unique(actual)) == 2:
        print(
            "Log loss: "
            f"{log_loss(actual, up_probabilities, labels=[0, 1]):.4f}"
        )
    else:
        print(
            "Log loss: not reported because the completed observations "
            "contain only one actual class."
        )

    print("\nRecent completed predictions:")
    print(
        completed[
            [
                "Signal_Date",
                "Outcome_Date",
                "Signal",
                "Up_Probability",
                "Actual",
                "Correct",
            ]
        ].tail(10).to_string(index=False)
    )

    print(
        "\nReminder: a small sample is not enough to establish "
        "reliable predictive performance."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Downloading {SYMBOL} market data...")

    raw_data = get_market_data(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
    )

    completed_data = get_completed_market_data(raw_data)

    print(
        "Latest eligible completed session:",
        completed_data.index[-1].strftime("%Y-%m-%d"),
    )

    # Calculate indicators on the eligible completed data only.
    data = calculate_indicators(completed_data)

    features = get_features_for_market(data)

    log = load_prediction_log()

    # Settle pending records first.
    log = update_previous_outcomes(log, data)

    # Record a new prediction only if this date has not been logged.
    log = generate_prediction(data, features, log)

    # Save all records, including settled outcomes.
    save_prediction_log(log)

    print(f"\nPrediction log saved to: {LOG_FILE}")

    print_evaluation(log)

    print(
        "\nPaper testing only: no real trades are placed. "
        "The model is retrained on available historical data "
        "for each new prediction."
    )


if __name__ == "__main__":
    main()
