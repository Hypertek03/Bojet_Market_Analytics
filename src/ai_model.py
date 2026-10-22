import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# ============================================================
# FEATURE GROUPS
# ============================================================

CORE_FEATURES = [
    "Price_MA10_Ratio",
    "RSI",
    "MACD_Difference",
    "Price_BB_Ratio",
    "BB_Position",
]

MOMENTUM_FEATURES = [
    "Daily_Return",
    "Return_5D",
    "Return_10D",
]

TREND_FEATURES = [
    "Price_MA20_Ratio",
    "Price_MA50_Ratio",
    "MA10_MA20_Ratio",
    "MA20_MA50_Ratio",
]

VOLATILITY_FEATURES = [
    "Volatility_10D",
]

VOLUME_FEATURES = [
    "Volume_Change",
    "Volume_MA20_Ratio",
]


# Features for markets with usable volume data.
FEATURES = (
    CORE_FEATURES
    + MOMENTUM_FEATURES
    + TREND_FEATURES
    + VOLATILITY_FEATURES
    + VOLUME_FEATURES
)

# Features for markets without usable volume data.
FEATURES_WITHOUT_VOLUME = (
    CORE_FEATURES
    + MOMENTUM_FEATURES
    + TREND_FEATURES
    + VOLATILITY_FEATURES
)


# ============================================================
# SELECT FEATURES FOR EACH MARKET
# ============================================================

def get_features_for_market(data):
    """
    Select features according to market data availability.

    Volume features are used only when the market has
    usable, non-zero volume data.
    """

    has_usable_volume = (
        "Volume" in data.columns
        and data["Volume"].notna().any()
        and (data["Volume"].fillna(0) > 0).any()
    )

    if has_usable_volume:
        return FEATURES.copy()

    return FEATURES_WITHOUT_VOLUME.copy()


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(data):
    """
    Create a target indicating whether the next closing
    price is higher than the current closing price.

    Target:
        1 = Next closing price is higher.
        0 = Next closing price is unchanged or lower.

    The final row has no known next-day target and is
    therefore excluded from model training.
    """

    data = data.copy()

    data["Next_Close"] = data["Close"].shift(-1)

    data["Target"] = pd.Series(
        pd.NA,
        index=data.index,
        dtype="Int64",
    )

    valid_target = data["Next_Close"].notna()

    data.loc[valid_target, "Target"] = (
        data.loc[valid_target, "Next_Close"]
        > data.loc[valid_target, "Close"]
    ).astype(int)

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):
    """
    Prepare features and targets for model training.
    """

    features = get_features_for_market(data)

    required_columns = features + [
        "Next_Close",
        "Target",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_data = data.dropna(
        subset=required_columns
    ).copy()

    if len(model_data) < 30:
        raise ValueError(
            "Not enough usable market data to train the model."
        )

    X = model_data[features]

    y = model_data["Target"].astype(int)

    return X, y


# ============================================================
# CREATE MODEL
# ============================================================

def build_model():
    """
    Create the Random Forest model used by the project.
    """

    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y):
    """
    Train on earlier observations and test on later ones.

    The data must already be ordered chronologically.
    """

    if len(X) < 30:
        raise ValueError(
            "Not enough samples to train and test the model."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "Feature and target indexes do not match."
        )

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one target class."
        )

    model = build_model()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        accuracy,
    )


# ============================================================
# TIME-SERIES VALIDATION
# ============================================================

def validate_model(X, y):
    """
    Evaluate the model using chronological validation folds.
    """

    if len(X) < 30:
        raise ValueError(
            "Not enough samples for time-series validation."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "Feature and target indexes do not match."
        )

    tscv = TimeSeriesSplit(n_splits=5)

    scores = []

    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        if y_train.nunique() < 2:
            continue

        model = build_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        score = accuracy_score(
            y_test,
            predictions,
        )

        scores.append(score)

    if not scores:
        raise ValueError(
            "Time-series validation could not produce a valid score."
        )

    average_accuracy = sum(scores) / len(scores)

    return scores, average_accuracy


# ============================================================
# PREDICT LATEST MARKET DIRECTION
# ============================================================

def predict_latest(model, latest_data):
    """
    Predict the next closing-price direction.

    Probabilities are model outputs and are not guaranteed
    to be calibrated probabilities of future market movements.
    """

    model_features = list(model.feature_names_in_)

    missing_features = [
        feature
        for feature in model_features
        if feature not in latest_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing prediction features: {missing_features}"
        )

    if latest_data.empty:
        raise ValueError(
            "No latest market data is available."
        )

    latest_features = latest_data[model_features]

    if latest_features.isna().any().any():
        raise ValueError(
            "Latest market data contains missing model features."
        )

    prediction = model.predict(
        latest_features
    )[0]

    probabilities = model.predict_proba(
        latest_features
    )[0]

    classes = list(model.classes_)

    up_probability = (
        float(probabilities[classes.index(1)]) * 100
        if 1 in classes
        else 0.0
    )

    down_probability = (
        float(probabilities[classes.index(0)]) * 100
        if 0 in classes
        else 0.0
    )

    signal = "UP" if prediction == 1 else "DOWN"

    highest_probability = max(
        up_probability,
        down_probability,
    )

    if highest_probability >= 85:
        confidence = "VERY HIGH"
    elif highest_probability >= 75:
        confidence = "HIGH"
    elif highest_probability >= 65:
        confidence = "MODERATE"
    elif highest_probability >= 55:
        confidence = "LOW"
    else:
        confidence = "VERY LOW"

    return {
        "signal": signal,
        "up_probability": up_probability,
        "down_probability": down_probability,
        "confidence": confidence,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model):
    """
    Return Random Forest feature importance in descending order.
    """

    feature_names = list(model.feature_names_in_)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    })

    importance_df = (
        importance_df.sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# COMPARE FEATURE SETS
# ============================================================

def compare_feature_sets(data):
    """
    Compare several feature groups using a chronological
    train/test split.
    """

    data = create_target(data)

    available_features = get_features_for_market(data)

    feature_sets = {
        "Core Indicators": CORE_FEATURES,
        "Core + Momentum": (
            CORE_FEATURES + MOMENTUM_FEATURES
        ),
        "Core + Trend": (
            CORE_FEATURES + TREND_FEATURES
        ),
        "All Features": available_features,
    }

    results = []

    for name, feature_list in feature_sets.items():
        if not all(
            feature in data.columns
            for feature in feature_list
        ):
            continue

        model_data = data.dropna(
            subset=feature_list + [
                "Next_Close",
                "Target",
            ]
        ).copy()

        if len(model_data) < 60:
            continue

        X = model_data[feature_list]

        y = model_data["Target"].astype(int)

        split_index = int(len(X) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        if y_train.nunique() < 2:
            continue

        model = build_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = (
            accuracy_score(y_test, predictions) * 100
        )

        baseline_accuracy = (
            y_test.value_counts(normalize=True).max() * 100
        )

        results.append({
            "Feature Set": name,
            "Features": len(feature_list),
            "Test Accuracy": accuracy,
            "Baseline Accuracy": baseline_accuracy,
        })

    return pd.DataFrame(results)