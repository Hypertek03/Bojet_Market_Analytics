import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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


def prepare_validation_data(data):
    """Prepare next-session intraday direction targets."""

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

    all_features = get_features_for_market(data)

    # Use only observations complete for BOTH feature sets.
    required = list(
        dict.fromkeys(
            all_features
            + ["Next_Open", "Next_Close", "Next_Date", "Target"]
        )
    )

    model_data = data.dropna(subset=required).copy()
    model_data["Target"] = model_data["Target"].astype(int)

    if len(model_data) < 100:
        raise ValueError(
            f"Not enough complete observations: {len(model_data)}"
        )

    missing_core = [
        feature
        for feature in CORE_FEATURES
        if feature not in model_data.columns
    ]

    if missing_core:
        raise ValueError(
            f"Missing core features: {missing_core}"
        )

    return model_data, all_features


def evaluate_feature_set(model_data, features, feature_set_name):
    """Evaluate one feature set with chronological validation."""

    X = model_data[features]
    y = model_data["Target"]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_records = []
    all_predictions = []
    all_actual = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() < 2:
            print(
                f"Skipping {feature_set_name}, fold {fold}: "
                "training data contains only one class."
            )
            continue

        model = build_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions,
        )

        majority_baseline = max(
            y_test.mean(),
            1 - y_test.mean(),
        )

        fold_records.append({
            "Feature Set": feature_set_name,
            "Fold": fold,
            "Test Start": model_data.index[test_idx[0]].date(),
            "Test End": model_data.index[test_idx[-1]].date(),
            "Train Samples": len(train_idx),
            "Test Samples": len(test_idx),
            "Accuracy (%)": accuracy * 100,
            "Balanced Accuracy (%)": balanced_accuracy * 100,
            "Majority Baseline (%)": majority_baseline * 100,
        })

        all_predictions.extend(predictions.tolist())
        all_actual.extend(y_test.tolist())

    if not fold_records:
        raise ValueError(
            f"No valid validation folds for {feature_set_name}."
        )

    overall_accuracy = accuracy_score(
        all_actual,
        all_predictions,
    )

    overall_balanced_accuracy = balanced_accuracy_score(
        all_actual,
        all_predictions,
    )

    return fold_records, {
        "Feature Set": feature_set_name,
        "Features": len(features),
        "Mean Fold Accuracy (%)": pd.DataFrame(
            fold_records
        )["Accuracy (%)"].mean(),
        "Overall Accuracy (%)": overall_accuracy * 100,
        "Overall Balanced Accuracy (%)": (
            overall_balanced_accuracy * 100
        ),
        "Predictions": len(all_predictions),
    }


def main():
    print(f"Downloading {SYMBOL} historical data...")

    data = get_market_data(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
    )

    data = calculate_indicators(data)

    model_data, all_features = prepare_validation_data(data)

    print(f"\nUsable observations: {len(model_data)}")
    print(f"Core features: {len(CORE_FEATURES)}")
    print(f"All available features: {len(all_features)}")
    print(f"Chronological folds: {N_SPLITS}")

    core_folds, core_summary = evaluate_feature_set(
        model_data,
        CORE_FEATURES,
        "Core 5",
    )

    all_folds, all_summary = evaluate_feature_set(
        model_data,
        all_features,
        "All available",
    )

    fold_results = pd.DataFrame(core_folds + all_folds)
    summary_results = pd.DataFrame([
        core_summary,
        all_summary,
    ])

    print("\nFOLD RESULTS")
    print("=" * 100)
    print(fold_results.round(2).to_string(index=False))

    print("\nSUMMARY")
    print("=" * 100)
    print(summary_results.round(2).to_string(index=False))

    print(
        "\nInterpretation: compare both models across the same folds. "
        "Accuracy and balanced accuracy describe classification, "
        "not investment returns."
    )

    print(
        "These results use historical data previously explored in "
        "this project; they are not an untouched final holdout."
    )


if __name__ == "__main__":
    main()
