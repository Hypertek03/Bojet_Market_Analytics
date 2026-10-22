import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.ai_model import build_model, get_features_for_market
from src.indicators import calculate_indicators
from src.market_data import get_market_data


def main():
    print("Downloading AAPL market data...")

    data = get_market_data("AAPL", period="2y", interval="1d")
    data = calculate_indicators(data)

    # The target is whether the NEXT session closes above its open.
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
        subset=features + ["Target", "Next_Date"]
    ).copy()

    model_data["Target"] = model_data["Target"].astype(int)

    X = model_data[features]
    y = model_data["Target"]

    if len(model_data) < 100:
        raise ValueError("Not enough observations for reliable diagnostics.")

    print(f"\nUsable observations: {len(model_data)}")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features}")

    print("\nACTUAL TARGET DISTRIBUTION")
    print("-" * 45)

    distribution = y.value_counts().sort_index()
    for label, count in distribution.items():
        name = "DOWN" if label == 0 else "UP"
        print(f"{name}: {count} ({count / len(y) * 100:.2f}%)")

    splitter = TimeSeriesSplit(n_splits=5)

    all_actual = []
    all_predictions = []
    all_majority_predictions = []
    all_up_predictions = []
    all_down_predictions = []

    fold_rows = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X), start=1
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = build_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test).astype(int)

        # Baselines are determined without looking at test labels.
        majority_class = int(y_train.value_counts().idxmax())
        majority_predictions = [majority_class] * len(y_test)
        up_predictions = [1] * len(y_test)
        down_predictions = [0] * len(y_test)

        fold_rows.append({
            "Fold": fold,
            "Train samples": len(train_index),
            "Test samples": len(test_index),
            "Test start": str(model_data.index[test_index[0]].date()),
            "Test end": str(model_data.index[test_index[-1]].date()),
            "AI accuracy (%)": accuracy_score(
                y_test, predictions
            ) * 100,
            "AI balanced accuracy (%)": balanced_accuracy_score(
                y_test, predictions
            ) * 100,
            "Majority baseline (%)": accuracy_score(
                y_test, majority_predictions
            ) * 100,
            "Always UP (%)": accuracy_score(
                y_test, up_predictions
            ) * 100,
            "Always DOWN (%)": accuracy_score(
                y_test, down_predictions
            ) * 100,
        })

        all_actual.extend(y_test.tolist())
        all_predictions.extend(predictions.tolist())
        all_majority_predictions.extend(majority_predictions)
        all_up_predictions.extend(up_predictions)
        all_down_predictions.extend(down_predictions)

    results = pd.DataFrame(fold_rows)

    print("\nWALK-FORWARD FOLD RESULTS")
    print("=" * 110)
    print(results.round(2).to_string(index=False))

    print("\nOVERALL OUT-OF-FOLD METRICS")
    print("=" * 65)

    comparisons = {
        "Random Forest": all_predictions,
        "Training majority baseline": all_majority_predictions,
        "Always UP": all_up_predictions,
        "Always DOWN": all_down_predictions,
    }

    for name, predictions in comparisons.items():
        accuracy = accuracy_score(all_actual, predictions) * 100
        balanced = balanced_accuracy_score(
            all_actual, predictions
        ) * 100

        print(
            f"{name:28s} "
            f"Accuracy: {accuracy:6.2f}% | "
            f"Balanced accuracy: {balanced:6.2f}%"
        )

    print("\nRANDOM FOREST CLASSIFICATION DETAILS")
    print("-" * 65)

    print(
        f"UP precision: "
        f"{precision_score(all_actual, all_predictions, zero_division=0):.3f}"
    )
    print(
        f"UP recall: "
        f"{recall_score(all_actual, all_predictions, zero_division=0):.3f}"
    )

    matrix = confusion_matrix(
        all_actual,
        all_predictions,
        labels=[0, 1],
    )

    print("\nConfusion matrix:")
    print("Rows = actual [DOWN, UP]")
    print("Columns = predicted [DOWN, UP]")
    print(matrix)

    actual_up = sum(all_actual)
    predicted_up = sum(all_predictions)

    print(f"\nActual UP outcomes: {actual_up}")
    print(f"Predicted UP outcomes: {predicted_up}")
    print(f"Actual DOWN outcomes: {len(all_actual) - actual_up}")
    print(f"Predicted DOWN outcomes: {len(all_predictions) - predicted_up}")

    print("\nInterpretation:")
    print(
        "Accuracy measures the percentage of correct predictions. "
        "Balanced accuracy gives equal weight to UP and DOWN recall."
    )
    print(
        "Compare the AI with the baselines before concluding that it "
        "has useful predictive power."
    )


if __name__ == "__main__":
    main()
