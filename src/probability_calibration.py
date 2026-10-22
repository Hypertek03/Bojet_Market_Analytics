import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit

from src.ai_model import (
    CORE_FEATURES,
    build_model,
    get_features_for_market,
)
from src.indicators import calculate_indicators
from src.market_data import get_market_data


SYMBOL = "AAPL"
PERIOD = "2y"
INTERVAL = "1d"
N_SPLITS = 5

CONFIDENCE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]


def prepare_data(data):
    """Create next-session intraday direction targets."""

    data = data.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")].copy()

    data["Next_Open"] = data["Open"].shift(-1)
    data["Next_Close"] = data["Close"].shift(-1)
    data["Next_Date"] = data.index.to_series().shift(-1)

    valid_target = (
        data["Next_Open"].notna()
        & data["Next_Close"].notna()
        & data["Next_Date"].notna()
    )

    data["Target"] = pd.Series(
        pd.NA,
        index=data.index,
        dtype="Int64",
    )

    data.loc[valid_target, "Target"] = (
        data.loc[valid_target, "Next_Close"]
        > data.loc[valid_target, "Next_Open"]
    ).astype(int)

    features = get_features_for_market(data)

    required = features + [
        "Next_Open",
        "Next_Close",
        "Next_Date",
        "Target",
    ]

    model_data = data.dropna(subset=required).copy()
    model_data["Target"] = model_data["Target"].astype(int)

    if len(model_data) < 100:
        raise ValueError(
            f"Not enough complete observations: {len(model_data)}"
        )

    return model_data, features


def generate_oof_predictions(model_data, features):
    """Generate out-of-fold probabilities chronologically."""

    X = model_data[features]
    y = model_data["Target"]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() < 2:
            print(f"Skipping fold {fold}: only one training class.")
            continue

        model = build_model()
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)

        if 1 not in classes:
            up_probabilities = np.zeros(len(X_test))
        else:
            up_probabilities = probabilities[:, classes.index(1)]

        if 0 not in classes:
            down_probabilities = np.zeros(len(X_test))
        else:
            down_probabilities = probabilities[:, classes.index(0)]

        predictions = (up_probabilities >= 0.5).astype(int)

        for position, actual, prediction, up_prob, down_prob in zip(
            test_idx,
            y_test.to_numpy(),
            predictions,
            up_probabilities,
            down_probabilities,
        ):
            records.append({
                "Date": model_data.index[position],
                "Next_Date": model_data.iloc[position]["Next_Date"],
                "Fold": fold,
                "Actual": int(actual),
                "Prediction": int(prediction),
                "Up_Probability": float(up_prob),
                "Down_Probability": float(down_prob),
                "Confidence": float(max(up_prob, down_prob)),
            })

    if not records:
        raise ValueError("No out-of-fold predictions were generated.")

    return pd.DataFrame(records).sort_values("Date").reset_index(drop=True)


def evaluate_probabilities(predictions):
    """Summarize overall classification and probability quality."""

    actual = predictions["Actual"].to_numpy()
    predicted = predictions["Prediction"].to_numpy()
    up_probabilities = predictions["Up_Probability"].to_numpy()

    majority_class = int(pd.Series(actual).mode().iloc[0])
    baseline_predictions = np.full(len(actual), majority_class)

    summary = {
        "OOF Predictions": len(predictions),
        "UP Actual (%)": actual.mean() * 100,
        "Accuracy (%)": accuracy_score(actual, predicted) * 100,
        "Balanced Accuracy (%)": (
            balanced_accuracy_score(actual, predicted) * 100
        ),
        "Majority Baseline (%)": (
            accuracy_score(actual, baseline_predictions) * 100
        ),
        "Brier Score": brier_score_loss(actual, up_probabilities),
        "Log Loss": log_loss(
            actual,
            np.column_stack([
                1 - up_probabilities,
                up_probabilities,
            ]),
            labels=[0, 1],
        ),
    }

    return summary


def evaluate_confidence_thresholds(predictions):
    """Evaluate high-confidence predictions without trading."""

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:
        selected = predictions[
            predictions["Confidence"] >= threshold
        ].copy()

        if selected.empty:
            rows.append({
                "Threshold (%)": threshold * 100,
                "Selected (%)": 0.0,
                "Predictions": 0,
                "Accuracy (%)": np.nan,
                "Balanced Accuracy (%)": np.nan,
                "Actual UP (%)": np.nan,
            })
            continue

        rows.append({
            "Threshold (%)": threshold * 100,
            "Selected (%)": (
                len(selected) / len(predictions) * 100
            ),
            "Predictions": len(selected),
            "Accuracy (%)": accuracy_score(
                selected["Actual"],
                selected["Prediction"],
            ) * 100,
            "Balanced Accuracy (%)": balanced_accuracy_score(
                selected["Actual"],
                selected["Prediction"],
            ) * 100,
            "Actual UP (%)": (
                selected["Actual"].mean() * 100
            ),
        })

    return pd.DataFrame(rows)


def evaluate_calibration_bins(predictions):
    """Compare predicted UP probabilities with observed UP rates."""

    actual = predictions["Actual"].to_numpy()
    probabilities = predictions["Up_Probability"].to_numpy()

    observed, predicted = calibration_curve(
        actual,
        probabilities,
        n_bins=5,
        strategy="uniform",
    )

    return pd.DataFrame({
        "Mean Predicted UP (%)": predicted * 100,
        "Observed UP (%)": observed * 100,
    })


def main():
    print(f"Downloading {SYMBOL} historical data...")

    data = get_market_data(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
    )

    data = calculate_indicators(data)

    model_data, features = prepare_data(data)

    print(f"Usable observations: {len(model_data)}")
    print(f"Features: {len(features)}")
    print(f"Validation folds: {N_SPLITS}")

    predictions = generate_oof_predictions(
        model_data,
        features,
    )

    summary = evaluate_probabilities(predictions)
    thresholds = evaluate_confidence_thresholds(predictions)
    calibration = evaluate_calibration_bins(predictions)

    print("\nOVERALL PROBABILITY EVALUATION")
    print("=" * 80)
    print(pd.Series(summary).round(4).to_string())

    print("\nCONFIDENCE THRESHOLD RESULTS")
    print("=" * 80)
    print(thresholds.round(2).to_string(index=False))

    print("\nPROBABILITY CALIBRATION BINS")
    print("=" * 80)
    print(calibration.round(2).to_string(index=False))

    print(
        "\nReminder: confidence thresholds are exploratory. "
        "They must be chosen using development data and then "
        "evaluated on separate future data."
    )

    print(
        "This script evaluates classification probabilities, "
        "not trading profitability after transaction costs."
    )


if __name__ == "__main__":
    main()
