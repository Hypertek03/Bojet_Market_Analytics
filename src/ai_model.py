import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# ==========================================
# FEATURE GROUPS
# ==========================================

CORE_FEATURES = [
    "Price_MA10_Ratio",
    "RSI",
    "MACD_Difference",
    "Price_BB_Ratio",
    "BB_Position"
]

MOMENTUM_FEATURES = [
    "Daily_Return",
    "Return_5D",
    "Return_10D"
]

TREND_FEATURES = [
    "Price_MA20_Ratio",
    "Price_MA50_Ratio",
    "MA10_MA20_Ratio",
    "MA20_MA50_Ratio"
]

VOLATILITY_VOLUME_FEATURES = [
    "Volatility_10D",
    "Volume_Change",
    "Volume_MA20_Ratio"
]


# ==========================================
# CURRENT BOJET AI FEATURES
# ==========================================

FEATURES = (
    CORE_FEATURES
    + MOMENTUM_FEATURES
    + TREND_FEATURES
    + VOLATILITY_VOLUME_FEATURES
)


# ==========================================
# CREATE TARGET
# ==========================================

def create_target(data):

    data = data.copy()

    data["Next_Close"] = (
        data["Close"].shift(-1)
    )

    data["Target"] = (
        data["Next_Close"]
        > data["Close"]
    ).astype(int)

    return data


# ==========================================
# PREPARE DATA
# ==========================================

def prepare_data(data):

    model_data = data.dropna(
        subset=FEATURES + ["Next_Close"]
    ).copy()

    X = model_data[FEATURES]

    y = model_data["Target"]

    return X, y


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model(X, y):

    split_index = int(
        len(X) * 0.8
    )

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        accuracy
    )


# ==========================================
# TIME SERIES VALIDATION
# ==========================================

def validate_model(X, y):

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    scores = []

    for train_index, test_index in tscv.split(X):

        X_train = X.iloc[train_index]

        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]

        y_test = y.iloc[test_index]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        score = accuracy_score(
            y_test,
            predictions
        )

        scores.append(score)

    average_accuracy = (
        sum(scores)
        / len(scores)
    )

    return (
        scores,
        average_accuracy
    )


# ==========================================
# LATEST PREDICTION
# ==========================================

def predict_latest(
    model,
    latest_data
):

    latest_features = (
        latest_data[FEATURES]
    )

    prediction = model.predict(
        latest_features
    )[0]

    probability = model.predict_proba(
        latest_features
    )[0]

    down_probability = (
        probability[0] * 100
    )

    up_probability = (
        probability[1] * 100
    )

    signal = (
        "UP"
        if prediction == 1
        else "DOWN"
    )

    highest_probability = max(
        up_probability,
        down_probability
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
        "confidence": confidence
    }


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def get_feature_importance(model):

    importance = (
        model.feature_importances_
    )

    feature_importance = []

    for feature, value in zip(
        FEATURES,
        importance
    ):

        feature_importance.append({
            "Feature": feature,
            "Importance": value
        })

    importance_df = pd.DataFrame(
        feature_importance
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


# ==========================================
# FEATURE SET EXPERIMENT
# ==========================================

def compare_feature_sets(data):

    data = create_target(data)

    feature_sets = {

        "Core Indicators": CORE_FEATURES,

        "Core + Momentum": (
            CORE_FEATURES
            + MOMENTUM_FEATURES
        ),

        "Core + Trend": (
            CORE_FEATURES
            + TREND_FEATURES
        ),

        "All Features": FEATURES
    }

    results = []

    for name, feature_list in feature_sets.items():

        model_data = data.dropna(
            subset=feature_list
            + ["Next_Close"]
        ).copy()

        X = model_data[feature_list]

        y = model_data["Target"]

        if len(X) < 60:

            continue

        split_index = int(
            len(X) * 0.8
        )

        X_train = X.iloc[:split_index]

        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]

        y_test = y.iloc[split_index:]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
            * 100
        )

        results.append({
            "Feature Set": name,
            "Features": len(feature_list),
            "Test Accuracy": accuracy
        })

    return pd.DataFrame(results)