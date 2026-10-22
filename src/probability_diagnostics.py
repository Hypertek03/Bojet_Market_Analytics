import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit

from src.ai_model import build_model, get_features_for_market
from src.indicators import calculate_indicators
from src.market_data import get_market_data


SYMBOL = "AAPL"
PERIOD = "2y"
INTERVAL = "1d"
N_SPLITS = 5
N_BINS = 5


def prepare_data(data):
    """Prepare features and next-session intraday targets."""

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
    """Generate chronological out-of-fold probabilities."""

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

        up_prob = (
            probabilities[:, classes.index(1)]
            if 1 in classes
            else np.zeros(len(X_test))
        )

        prediction = (up_prob >= 0.5).astype(int)

        # Use only the training period to estimate the baseline UP rate.
        training_up_rate = float(y_train.mean())

        for j, position in enumerate(test_idx):
            records.append({
                "Date": model_data.index[position],
                "Next_Date": model_data.iloc[position]["Next_Date"],
                "Fold": fold,
                "Actual": int(y_test.iloc[j]),
                "Prediction": int(prediction[j]),
                "Up_Probability": float(up_prob[j]),
                "Baseline_UP_Probability": training_up_rate,
            })

    if not records:
        raise ValueError("No out-of-fold predictions were generated.")

    return pd.DataFrame(records).sort_values("Date").reset_index(drop=True)


def safe_log_loss(actual, probabilities):
    """Calculate binary log loss with probabilities clipped for safety."""

    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    return log_loss(
        actual,
        np.column_stack([1 - probabilities, probabilities]),
        labels=[0, 1],
    )


def evaluate_overall(predictions):
    """Compare model probabilities with the training-rate baseline."""

    actual = predictions["Actual"].to_numpy()
    up_prob = predictions["Up_Probability"].to_numpy()
    baseline_prob = predictions["Baseline_UP_Probability"].to_numpy()

    model_brier = brier_score_loss(actual, up_prob)
    baseline_brier = brier_score_loss(actual, baseline_prob)

    model_log_loss = safe_log_loss(actual, up_prob)
    baseline_log_loss = safe_log_loss(actual, baseline_prob)

    majority_class = int(pd.Series(actual).mode().iloc[0])
    majority_predictions = np.full(len(actual), majority_class)

    return pd.DataFrame([{
        "Predictions": len(actual),
        "Actual UP (%)": actual.mean() * 100,
        "Model Accuracy (%)": accuracy_score(
            actual,
            predictions["Prediction"],
        ) * 100,
        "Balanced Accuracy (%)": balanced_accuracy_score(
            actual,
            predictions["Prediction"],
        ) * 100,
        "Majority Baseline Accuracy (%)": accuracy_score(
            actual,
            majority_predictions,
        ) * 100,
        "Model Brier": model_brier,
        "Training-Rate Baseline Brier": baseline_brier,
        "Brier Skill Score (%)": (
            (1 - model_brier / baseline_brier) * 100
            if baseline_brier > 0
            else np.nan
        ),
        "Model Log Loss": model_log_loss,
        "Training-Rate Baseline Log Loss": baseline_log_loss,
    }])


def evaluate_by_fold(predictions):
    """Evaluate probability quality separately for each fold."""

    rows = []

    for fold, group in predictions.groupby("Fold", sort=True):
        actual = group["Actual"].to_numpy()
        up_prob = group["Up_Probability"].to_numpy()
        baseline_prob = group["Baseline_UP_Probability"].to_numpy()

        model_brier = brier_score_loss(actual, up_prob)
        baseline_brier = brier_score_loss(actual, baseline_prob)

        rows.append({
            "Fold": fold,
            "Predictions": len(group),
            "Actual UP (%)": actual.mean() * 100,
            "Model Accuracy (%)": accuracy_score(
                actual,
                group["Prediction"],
            ) * 100,
            "Model Brier": model_brier,
            "Baseline Brier": baseline_brier,
            "Brier Skill Score (%)": (
                (1 - model_brier / baseline_brier) * 100
                if baseline_brier > 0
                else np.nan
            ),
            "Model Log Loss": safe_log_loss(actual, up_prob),
            "Baseline Log Loss": safe_log_loss(actual, baseline_prob),
        })

    return pd.DataFrame(rows)


def evaluate_calibration_bins(predictions):
    """Show prediction counts alongside observed UP frequencies."""

    bins = np.linspace(0, 1, N_BINS + 1)

    result = predictions.copy()

    result["Probability Bin"] = pd.cut(
        result["Up_Probability"],
        bins=bins,
        include_lowest=True,
    )

    rows = []

    for probability_bin, group in result.groupby(
        "Probability Bin",
        observed=False,
    ):
        if group.empty:
            continue

        rows.append({
            "Probability Bin": str(probability_bin),
            "Predictions": len(group),
            "Mean Predicted UP (%)": (
                group["Up_Probability"].mean() * 100
            ),
            "Observed UP (%)": group["Actual"].mean() * 100,
        })

    return pd.DataFrame(rows)


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

    overall = evaluate_overall(predictions)
    by_fold = evaluate_by_fold(predictions)
    calibration = evaluate_calibration_bins(predictions)

    print("\nOVERALL PROBABILITY COMPARISON")
    print("=" * 110)
    print(overall.round(4).to_string(index=False))

    print("\nRESULTS BY FOLD")
    print("=" * 110)
    print(by_fold.round(4).to_string(index=False))

    print("\nCALIBRATION BINS")
    print("=" * 110)
    print(calibration.round(2).to_string(index=False))

    print(
        "\nInterpretation: a positive Brier skill score means the model "
        "had a lower Brier score than the training-rate baseline."
    )
    print(
        "A negative score means the model's probability predictions "
        "were worse than that baseline on these observations."
    )
    print(
        "These are historical research results, not evidence of "
        "future profitability or an untouched final test."
    )


if __name__ == "__main__":
    main()
