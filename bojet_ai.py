import pandas as pd

from market_data import get_market_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# GET MARKET DATA
# ============================================================

data = get_market_data("AAPL")


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

# Moving Average
data["MA10"] = data["Close"].rolling(10).mean()


# RSI
delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

data["RSI"] = 100 - (100 / (1 + rs))


# MACD
ema12 = data["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = data["Close"].ewm(
    span=26,
    adjust=False
).mean()

data["MACD"] = ema12 - ema26

data["Signal"] = data["MACD"].ewm(
    span=9,
    adjust=False
).mean()


# Bollinger Bands
data["BB_Middle"] = data["Close"].rolling(20).mean()

bb_std = data["Close"].rolling(20).std()

data["BB_Upper"] = (
    data["BB_Middle"] + (2 * bb_std)
)

data["BB_Lower"] = (
    data["BB_Middle"] - (2 * bb_std)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

data["Price_MA10_Ratio"] = (
    data["Close"] / data["MA10"]
)

data["MACD_Difference"] = (
    data["MACD"] - data["Signal"]
)

data["Price_BB_Ratio"] = (
    data["Close"] / data["BB_Middle"]
)

data["BB_Position"] = (
    (data["Close"] - data["BB_Lower"])
    / (data["BB_Upper"] - data["BB_Lower"])
)


# ============================================================
# MARKET FEATURES
# ============================================================

data["Daily_Return"] = data["Close"].pct_change()

data["Previous_Return"] = (
    data["Daily_Return"].shift(1)
)

data["Return_5D"] = data["Close"].pct_change(5)

data["Return_10D"] = data["Close"].pct_change(10)

data["Volatility_10D"] = (
    data["Daily_Return"].rolling(10).std()
)

data["Volume_Change"] = (
    data["Volume"].pct_change()
)

data["MA10_Slope"] = (
    data["MA10"].pct_change()
)

data["Price_MA10_Distance"] = (
    (data["Close"] - data["MA10"])
    / data["MA10"]
)


# ============================================================
# LAG FEATURES
# ============================================================

data["Return_Lag1"] = (
    data["Daily_Return"].shift(1)
)

data["Return_Lag2"] = (
    data["Daily_Return"].shift(2)
)

data["Return_Lag3"] = (
    data["Daily_Return"].shift(3)
)

data["Return_Lag5"] = (
    data["Daily_Return"].shift(5)
)

data["RSI_Lag1"] = (
    data["RSI"].shift(1)
)

data["MACD_Difference_Lag1"] = (
    data["MACD_Difference"].shift(1)
)


# ============================================================
# TARGET
# ============================================================

data["Tomorrow_Close"] = (
    data["Close"].shift(-1)
)

data["Target"] = (
    data["Tomorrow_Close"] > data["Close"]
).astype(int)


# ============================================================
# SEPARATE FORECAST DATA FROM TRAINING DATA
# ============================================================

forecast_data = data.iloc[-1].copy()

training_data = (
    data.iloc[:-1]
    .dropna()
    .copy()
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()

print("TARGET DISTRIBUTION")
print("=" * 50)

print(
    training_data["Target"]
    .value_counts()
    .sort_index()
)

print()

print(
    training_data["Target"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


# ============================================================
# SELECTED FEATURES
# ============================================================

features = [
    "MACD_Difference_Lag1",
    "Daily_Return",
    "Return_Lag2",
    "MACD_Difference",
    "Return_10D",
    "Price_BB_Ratio",
    "Volume_Change",
    "Return_Lag1",
    "Price_MA10_Distance",
    "RSI_Lag1"
]


X = training_data[features]

y = training_data["Target"]


# ============================================================
# 80/20 TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split = int(
    len(training_data) * 0.80
)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(
    ascending=False
)

print()

print("FEATURE IMPORTANCE")
print("=" * 50)

for feature, value in importance.items():
    print(
        f"{feature:25} "
        f"{value:.4f}"
    )


# ============================================================
# TIME SERIES CROSS-VALIDATION
# ============================================================

tscv = TimeSeriesSplit(
    n_splits=5
)

cv_scores = []

cv_baseline_scores = []


for fold, (
    train_index,
    test_index
) in enumerate(
    tscv.split(X),
    start=1
):

    X_train_cv = X.iloc[train_index]
    X_test_cv = X.iloc[test_index]

    y_train_cv = y.iloc[train_index]
    y_test_cv = y.iloc[test_index]


    cv_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    cv_model.fit(
        X_train_cv,
        y_train_cv
    )

    cv_predictions = cv_model.predict(
        X_test_cv
    )

    cv_accuracy = accuracy_score(
        y_test_cv,
        cv_predictions
    )


    # Majority-class baseline
    cv_majority_class = (
        y_train_cv.mode()[0]
    )

    cv_baseline_predictions = [
        cv_majority_class
    ] * len(y_test_cv)

    cv_baseline_accuracy = (
        accuracy_score(
            y_test_cv,
            cv_baseline_predictions
        )
    )


    cv_scores.append(
        cv_accuracy
    )

    cv_baseline_scores.append(
        cv_baseline_accuracy
    )


    print(
        f"Fold {fold}: "
        f"Model = "
        f"{cv_accuracy * 100:.2f}% | "
        f"Baseline = "
        f"{cv_baseline_accuracy * 100:.2f}%"
    )


print()

print(
    f"Average CV model accuracy: "
    f"{sum(cv_scores) / len(cv_scores) * 100:.2f}%"
)

print(
    f"Average CV baseline accuracy: "
    f"{sum(cv_baseline_scores) / len(cv_baseline_scores) * 100:.2f}%"
)


# ============================================================
# MODEL EVALUATION
# ============================================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

matrix = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# BASELINE
# ============================================================

majority_class = (
    y_train.mode()[0]
)

baseline_predictions = [
    majority_class
] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


# ============================================================
# MODEL RESULTS
# ============================================================

print()

print("BOJET AI MODEL")
print("=" * 50)

print(
    f"Total samples: "
    f"{len(training_data)}"
)

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)

print()

print(
    f"Model accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)

print()

print("Confusion Matrix:")
print(matrix)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

print()

print("CLASSIFICATION METRICS")
print("=" * 50)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall:    "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score:  "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

permutation = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

permutation_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": permutation.importances_mean
})

permutation_importance_df = (
    permutation_importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)


print()

print("PERMUTATION IMPORTANCE")
print("=" * 50)

for _, row in (
    permutation_importance_df.iterrows()
):

    print(
        f"{row['Feature']:25} "
        f"{row['Importance']:.4f}"
    )


# ============================================================
# LATEST MARKET DATA
# ============================================================

print()

print("=" * 60)
print("LATEST DATA AVAILABLE TO BOJET AI")
print("=" * 60)

print(
    data[["Close"]].tail()
)


# ============================================================
# BOJET AI FORECAST
# ============================================================

latest = forecast_data

latest_features = (
    latest[features]
    .to_frame()
    .T
)

probabilities = (
    model.predict_proba(
        latest_features
    )[0]
)

down_probability = (
    probabilities[0] * 100
)

up_probability = (
    probabilities[1] * 100
)

prediction = model.predict(
    latest_features
)[0]


# ============================================================
# FORECAST OUTPUT
# ============================================================

print()

print("=" * 60)
print("              BOJET AI FORECAST")
print("=" * 60)

print()

print(
    f"Forecast Date: "
    f"{latest.name}"
)

print(
    f"Latest Price:  "
    f"${latest['Close']:.2f}"
)

print()

print("MODEL INPUTS")
print("-" * 60)

print(
    f"Previous_Return: "
    f"{latest['Previous_Return']:.4f}"
)

print(
    f"MACD_Difference: "
    f"{latest['MACD_Difference']:.4f}"
)

print(
    f"Volume_Change: "
    f"{latest['Volume_Change']:.4f}"
)

print(
    f"Return_Lag2: "
    f"{latest['Return_Lag2']:.4f}"
)

print(
    f"Return_Lag1: "
    f"{latest['Return_Lag1']:.4f}"
)

print(
    f"RSI_Lag1: "
    f"{latest['RSI_Lag1']:.4f}"
)

print(
    f"MACD_Difference_Lag1: "
    f"{latest['MACD_Difference_Lag1']:.4f}"
)

print(
    f"Daily_Return: "
    f"{latest['Daily_Return']:.4f}"
)

print(
    f"Price_MA10_Distance: "
    f"{latest['Price_MA10_Distance']:.4f}"
)

print(
    f"Volatility_10D: "
    f"{latest['Volatility_10D']:.4f}"
)

print(
    f"MA10_Slope: "
    f"{latest['MA10_Slope']:.4f}"
)

print(
    f"Return_10D: "
    f"{latest['Return_10D']:.4f}"
)

print(
    f"Return_Lag3: "
    f"{latest['Return_Lag3']:.4f}"
)

print(
    f"Price_MA10_Ratio: "
    f"{latest['Price_MA10_Ratio']:.4f}"
)

print(
    f"Price_BB_Ratio: "
    f"{latest['Price_BB_Ratio']:.4f}"
)


print()

print("MODEL PROBABILITIES")
print("-" * 60)

print(
    f"UP Probability:   "
    f"{up_probability:.2f}%"
)

print(
    f"DOWN Probability: "
    f"{down_probability:.2f}%"
)


print()

print("AI PREDICTION")
print("-" * 60)

if prediction == 1:
    print("Prediction: UP")
else:
    print("Prediction: DOWN")

print("=" * 60)