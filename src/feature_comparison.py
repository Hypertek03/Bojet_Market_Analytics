import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.ai_model import build_model, get_features_for_market
from src.indicators import calculate_indicators
from src.market_data import get_market_data


CORE_FEATURES = [
    "Price_MA10_Ratio",
    "RSI",
    "MACD_Difference",
    "Price_BB_Ratio",
    "BB_Position",
]


def main():
    print("Downloading AAPL market data...")

    data = get_market_data("AAPL", period="2y", interval="1d")
    data = calculate_indicators(data)

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

    all_features = get_features_for_market(data)

    # Use identical observations for both feature groups.
    model_data = data.dropna(
        subset=all_features + ["Target", "Next_Date"]
    ).copy()

    model_data["Target"] = model_data["Target"].astype(int)

    feature_groups = {
        "Core 5 features": CORE_FEATURES,
        "All available features": all_features,
    }

    splitter = TimeSeriesSplit(n_splits=5)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(model_data), start=1
    ):
        train = model_data.iloc[train_idx]
        test = model_data.iloc[test_idx]

        for group_name, features in feature_groups.items():
            model = build_model()
            model.fit(train[features], train["Target"])

            predictions = model.predict(test[features])
            actual = test["Target"]

            majority_class = int(train["Target"].mode().iloc[0])
            baseline = [majority_class] * len(test)

            fold_results.append({
                "Fold": fold,
                "Feature group": group_name,
                "Accuracy (%)": accuracy_score(
                    actual, predictions
                ) * 100,
                "Balanced accuracy (%)": balanced_accuracy_score(
                    actual, predictions
                ) * 100,
                "Majority baseline (%)": accuracy_score(
                    actual, baseline
                ) * 100,
            })

    results = pd.DataFrame(fold_results)

    print("\nRESULTS BY FOLD")
    print("=" * 95)
    print(results.round(2).to_string(index=False))

    print("\nAVERAGE RESULTS")
    print("=" * 75)

    summary = results.groupby("Feature group").agg(
        Mean_accuracy=("Accuracy (%)", "mean"),
        Mean_balanced_accuracy=("Balanced accuracy (%)", "mean"),
        Mean_baseline=("Majority baseline (%)", "mean"),
        Accuracy_std=("Accuracy (%)", "std"),
    )

    print(summary.round(2).to_string())

    print(
        "\nNote: These are historical validation results. "
        "A small accuracy difference may be due to chance."
    )


if __name__ == "__main__":
    main()
