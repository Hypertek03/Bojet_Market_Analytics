import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# GET MARKET DATA
# =========================

ticker = yf.Ticker("AAPL")

# Get 1 year of historical data
data = ticker.history(period="1y")


# =========================
# MOVING AVERAGE
# =========================

data["MA10"] = data["Close"].rolling(window=10).mean()


# =========================
# RSI
# =========================

delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

rs = average_gain / average_loss

data["RSI"] = 100 - (100 / (1 + rs))


# =========================
# MACD
# =========================

ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()

data["MACD"] = ema12 - ema26

data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()


# =========================
# BOLLINGER BANDS
# =========================

data["BB_Middle"] = data["Close"].rolling(window=20).mean()

data["BB_Std"] = data["Close"].rolling(window=20).std()

data["BB_Upper"] = data["BB_Middle"] + (2 * data["BB_Std"])

data["BB_Lower"] = data["BB_Middle"] - (2 * data["BB_Std"])


# =========================
# MACHINE LEARNING TARGET
# =========================

# Get the next day's closing price
data["Next_Close"] = data["Close"].shift(-1)

# 1 = next day's price is higher
# 0 = next day's price is not higher
data["Target"] = (
    data["Next_Close"] > data["Close"]
).astype(int)


# =========================
# IMPROVED ML FEATURES
# =========================

# How far the price is from the 10-day average
data["Price_MA10_Ratio"] = (
    data["Close"] / data["MA10"]
)

# How far the price is from the Bollinger middle band
data["Price_BB_Ratio"] = (
    data["Close"] / data["BB_Middle"]
)

# Position of price inside the Bollinger Band range
data["BB_Position"] = (
    (data["Close"] - data["BB_Lower"])
    /
    (data["BB_Upper"] - data["BB_Lower"])
)

# Difference between MACD and Signal
data["MACD_Difference"] = (
    data["MACD"] - data["Signal"]
)

# Daily percentage change
data["Daily_Return"] = data["Close"].pct_change()

# 5-day price momentum
data["Momentum_5D"] = data["Close"].pct_change(periods=5)

# 20-day volatility
data["Volatility_20D"] = data["Daily_Return"].rolling(
    window=20
).std()

# =========================
# REMOVE MISSING VALUES
# =========================

data = data.dropna()


# =========================
# ML FEATURES
# =========================

features = [
    "Price_MA10_Ratio",
    "RSI",
    "MACD_Difference",
    "Price_BB_Ratio",
    "BB_Position",
]

X = data[features]

y = data["Target"]


# =========================
# TRAIN / TEST SPLIT
# =========================

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


# =========================
# BOJET ML MODEL
# =========================

print("================================")
print("     BOJET ML MODEL")
print("================================")

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")


# =========================
# RANDOM FOREST MODEL
# =========================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print()
print("Random Forest model trained successfully!")

# =========================
# BOJET CALIBRATED MODEL
# =========================

calibrated_model = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    method="sigmoid",
    cv=TimeSeriesSplit(n_splits=5)
)

calibrated_model.fit(X_train, y_train)

calibrated_predictions = calibrated_model.predict(X_test)

calibrated_probabilities = (
    calibrated_model.predict_proba(X_test)
)

print()
print("================================")
print("   BOJET CALIBRATED MODEL")
print("================================")

calibrated_accuracy = accuracy_score(
    y_test,
    calibrated_predictions
)

print(
    f"Calibrated Accuracy: "
    f"{calibrated_accuracy * 100:.2f}%"
)

print("================================")

# =========================
# FEATURE IMPORTANCE
# =========================

print()
print("================================")
print("     BOJET FEATURE IMPORTANCE")
print("================================")

feature_importance = model.feature_importances_

for feature, importance in zip(
    features,
    feature_importance
):

    print(
        f"{feature}: "
        f"{importance * 100:.2f}%"
    )

print("================================")

# =========================
# TIME-SERIES VALIDATION
# =========================

print()
print("================================")
print("   BOJET TIME-SERIES VALIDATION")
print("================================")

tscv = TimeSeriesSplit(n_splits=5)

fold_scores = []

for fold, (train_index, test_index) in enumerate(
    tscv.split(X),
    start=1
):

    X_train_fold = X.iloc[train_index]
    X_test_fold = X.iloc[test_index]

    y_train_fold = y.iloc[train_index]
    y_test_fold = y.iloc[test_index]

    validation_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    validation_model.fit(
        X_train_fold,
        y_train_fold
    )

    fold_predictions = validation_model.predict(
        X_test_fold
    )

    fold_accuracy = accuracy_score(
        y_test_fold,
        fold_predictions
    )

    fold_scores.append(fold_accuracy)

    print(
        f"Fold {fold} Accuracy: "
        f"{fold_accuracy * 100:.2f}%"
    )

average_accuracy = sum(fold_scores) / len(fold_scores)

print()
print(
    f"Average Validation Accuracy: "
    f"{average_accuracy * 100:.2f}%"
)

print("================================")

# =========================
# MODEL PREDICTIONS
# =========================

predictions = model.predict(X_test)

print()
print("First 10 predictions:")

for prediction in predictions[:10]:

    if prediction == 1:
        print("UP")
    else:
        print("DOWN")

# =========================
# BOJET PROBABILITY COMPARISON
# =========================

original_probabilities = model.predict_proba(X_test)

calibrated_probabilities = calibrated_model.predict_proba(X_test)

print()
print("================================")
print(" BOJET PROBABILITY COMPARISON")
print("================================")

for i in range(10):

    original_up = original_probabilities[i][1] * 100
    calibrated_up = calibrated_probabilities[i][1] * 100

    print(
        f"Prediction {i + 1}: "
        f"Original UP = {original_up:.2f}% | "
        f"Calibrated UP = {calibrated_up:.2f}%"
    )

print("================================")

# =========================
# PREDICTION PROBABILITIES
# =========================

probabilities = model.predict_proba(X_test)

print()
print("BOJET AI FORECAST")
print("-------------------------")

for i in range(10):

    down_probability = probabilities[i][0] * 100

    up_probability = probabilities[i][1] * 100

    if predictions[i] == 1:
        prediction = "UP"
    else:
        prediction = "DOWN"

    print(f"Prediction: {prediction}")

    print(
        f"UP probability: "
        f"{up_probability:.2f}%"
    )

    print(
        f"DOWN probability: "
        f"{down_probability:.2f}%"
    )

    print()

# =========================
# BOJET CONFIDENCE ANALYSIS
# =========================

print()
print("================================")
print("   BOJET CONFIDENCE ANALYSIS")
print("================================")

test_probabilities = model.predict_proba(X_test)

confidence = test_probabilities.max(axis=1)

confidence_prediction = test_probabilities.argmax(axis=1)

confidence_correct = (
    confidence_prediction == y_test.to_numpy()
)

confidence_levels = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]

for lower, upper in confidence_levels:

    mask = (
        (confidence >= lower)
        &
        (confidence < upper)
    )

    count = mask.sum()

    if count > 0:

        accuracy = (
            confidence_correct[mask].mean()
        )

        print(
            f"{lower * 100:.0f}% - "
            f"{(upper if upper <= 1 else 1) * 100:.0f}%: "
            f"{accuracy * 100:.2f}% accuracy "
            f"({count} predictions)"
        )

    else:

        print(
            f"{lower * 100:.0f}% - "
            f"{(upper if upper <= 1 else 1) * 100:.0f}%: "
            f"No predictions"
        )

print("================================")

# =========================
# MODEL ACCURACY
# =========================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print(
    f"Model Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()

print("CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions
    )
)

print()

print("CONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================
# LATEST MARKET PREDICTION
# =========================

latest_features = X.iloc[[-1]]

latest_prediction = model.predict(
    latest_features
)[0]

latest_probability = model.predict_proba(
    latest_features
)[0]

up_probability = latest_probability[1] * 100

down_probability = latest_probability[0] * 100

# =========================
# BOJET AI CONFIDENCE
# =========================

highest_probability = max(
    up_probability,
    down_probability
)

if highest_probability >= 85:
    confidence_level = "VERY HIGH"

elif highest_probability >= 75:
    confidence_level = "HIGH"

elif highest_probability >= 65:
    confidence_level = "MODERATE"

elif highest_probability >= 55:
    confidence_level = "LOW"

else:
    confidence_level = "VERY LOW"

if latest_prediction == 1:
    latest_signal = "UP"
else:
    latest_signal = "DOWN"


# =========================
# LATEST TECHNICAL DATA
# =========================

latest_price = data["Close"].iloc[-1]

latest_rsi = data["RSI"].iloc[-1]

latest_macd = data["MACD"].iloc[-1]

latest_signal_line = data["Signal"].iloc[-1]

latest_ma10 = data["MA10"].iloc[-1]

latest_upper_band = data["BB_Upper"].iloc[-1]

latest_lower_band = data["BB_Lower"].iloc[-1]


# =========================
# DETERMINE TREND
# =========================

if latest_price > latest_ma10:
    trend = "BULLISH"

elif latest_price < latest_ma10:
    trend = "BEARISH"

else:
    trend = "NEUTRAL"


# =========================
# BOLLINGER BAND SIGNAL
# =========================

if latest_price > latest_upper_band:
    bollinger_signal = "ABOVE UPPER BAND"

elif latest_price < latest_lower_band:
    bollinger_signal = "BELOW LOWER BAND"

else:
    bollinger_signal = "WITHIN BANDS"


# =========================
# BOJET LATEST FORECAST
# =========================

print()
print("================================")
print("      BOJET LATEST FORECAST")
print("================================")

print("Asset: AAPL")

print()
print(
    f"Current Price: "
    f"${latest_price:.2f}"
)

print()

print(
    f"AI Prediction: "
    f"{latest_signal}"
)

print(
    f"UP Probability: "
    f"{up_probability:.2f}%"
)

print(
    f"DOWN Probability: "
    f"{down_probability:.2f}%"
)

print(f"AI Confidence: {confidence_level}")

print()
print("TECHNICAL ANALYSIS")
print("-------------------------")

print(
    f"10-Day Trend: "
    f"{trend}"
)

print(
    f"RSI: "
    f"{latest_rsi:.2f}"
)

print(
    f"MACD: "
    f"{latest_macd:.2f}"
)

print(
    f"Signal Line: "
    f"{latest_signal_line:.2f}"
)

print(
    f"10-Day Average: "
    f"${latest_ma10:.2f}"
)

print("================================")


# =========================
# BOJET DECISION ENGINE
# =========================

# Start with a neutral score
score = 0


# =========================
# AI CONFIDENCE SCORE
# =========================

if confidence_level == "VERY HIGH":

    if latest_prediction == 1:
        score += 3
    else:
        score -= 3

elif confidence_level == "HIGH":

    if latest_prediction == 1:
        score += 2
    else:
        score -= 2

elif confidence_level == "MODERATE":

    if latest_prediction == 1:
        score += 1
    else:
        score -= 1

elif confidence_level == "LOW":

    if latest_prediction == 1:
        score += 0.5
    else:
        score -= 0.5

else:

    # VERY LOW confidence
    score += 0


# -------------------------
# TREND SIGNAL
# -------------------------

if trend == "BULLISH":
    score += 1

elif trend == "BEARISH":
    score -= 1


# -------------------------
# RSI SIGNAL
# -------------------------

if latest_rsi < 30:
    score += 1

elif latest_rsi > 70:
    score -= 1


# -------------------------
# MACD SIGNAL
# -------------------------

if latest_macd > latest_signal_line:
    score += 1

else:
    score -= 1


# -------------------------
# BOLLINGER BAND SIGNAL
# -------------------------

if bollinger_signal == "BELOW LOWER BAND":
    score += 1

elif bollinger_signal == "ABOVE UPPER BAND":
    score -= 1


# =========================
# OVERALL SIGNAL
# =========================

if score >= 3:
    overall_signal = "BULLISH"

elif score <= -3:
    overall_signal = "BEARISH"

else:
    overall_signal = "NEUTRAL"


# =========================
# DISPLAY DECISION
# =========================

print()
print("================================")
print("     BOJET DECISION ENGINE")
print("================================")

print(
    f"AI Signal: "
    f"{latest_signal}"
)

print(
    f"Trend: "
    f"{trend}"
)

print(
    f"RSI: "
    f"{latest_rsi:.2f}"
)

if latest_macd > latest_signal_line:
    print("MACD Signal: BULLISH")
else:
    print("MACD Signal: BEARISH")

print(
    f"Bollinger Signal: "
    f"{bollinger_signal}"
)

print()
print(
    f"BOJET SCORE: "
    f"{score}"
)

print(
    f"OVERALL SIGNAL: "
    f"{overall_signal}"
)

print("================================")
