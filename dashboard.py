import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

from src.indicators import calculate_indicators

from src.ai_model import (
    create_target,
    prepare_data,
    train_model,
    validate_model,
    predict_latest,
    get_feature_importance,
    compare_feature_sets
)

from src.backtester import (
    backtest_strategy,
    backtest_buy_and_hold
)

from src.ai_backtester import (
    backtest_ai_strategy
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bojet Market Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(37, 99, 235, 0.08),
            transparent 35%
        ),
        #020617;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
    color: #f8fafc;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 30px;
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(
        145deg,
        rgba(15, 23, 42, 0.98),
        rgba(30, 41, 59, 0.92)
    );

    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 20px;
    min-height: 135px;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.18);
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
}

.metric-sub {
    color: #64748b;
    font-size: 13px;
    margin-top: 5px;
}

.ai-card {
    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.98),
            rgba(15, 23, 42, 0.98)
        );

    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 18px;
    padding: 25px;
    text-align: center;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);
}

.ai-title {
    color: #94a3b8;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
}

.ai-signal {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: 1px;
}

.ai-signal.bullish {
    color: #22c55e;
}

.ai-signal.bearish {
    color: #ef4444;
}

.ai-confidence {
    color: #cbd5e1;
    margin-top: 8px;
}

.prob-card {
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    margin-top: 15px;
}

.prob-title {
    color: #94a3b8;
    font-size: 13px;
}

.prob-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

.decision-card {
    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.98),
            rgba(30, 41, 59, 0.95)
        );

    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 18px;
    padding: 28px;
    text-align: center;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);
}

.decision-label {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
}

.decision-score {
    color: #f8fafc;
    font-size: 50px;
    font-weight: 900;
    margin: 5px 0;
}

.decision-signal {
    font-size: 25px;
    font-weight: 800;
}

.decision-signal.bullish {
    color: #22c55e;
}

.decision-signal.bearish {
    color: #ef4444;
}

.decision-signal.neutral {
    color: #f59e0b;
}

.info-box {
    background: rgba(15, 23, 42, 0.75);
    border-left: 4px solid #3b82f6;
    padding: 15px 18px;
    border-radius: 8px;
    color: #cbd5e1;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding-top: 40px;
    padding-bottom: 20px;
    font-size: 13px;
}

</style>
""")


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="main-title">
    📊 Bojet Market Analytics
</div>

<div class="main-subtitle">
    AI-powered market intelligence, technical analysis and
    historical strategy testing.
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Market Settings")

market_type = st.sidebar.selectbox(
    "Market Type",
    [
        "Stock",
        "Forex",
        "Crypto"
    ]
)


# ============================================================
# ASSETS
# ============================================================

stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META"
}

forex = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X"
}

crypto = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD"
}


if market_type == "Stock":

    assets = stocks

elif market_type == "Forex":

    assets = forex

else:

    assets = crypto


asset_name = st.sidebar.selectbox(
    "Asset",
    list(assets.keys())
)

ticker_symbol = assets[asset_name]


# ============================================================
# PERIOD
# ============================================================

period = st.sidebar.selectbox(
    "Historical Data",
    [
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y"
    ],
    index=3
)


# ============================================================
# CURRENCY
# ============================================================

if market_type == "Forex":

    currency_symbol = ""

else:

    currency_symbol = "$"


# ============================================================
# DOWNLOAD DATA
# ============================================================

try:

    data = yf.download(
        ticker_symbol,
        period=period,
        auto_adjust=True,
        progress=False
    )

except Exception as error:

    st.error(
        f"Unable to download market data: {error}"
    )

    st.stop()


if data.empty:

    st.error(
        "No market data was returned for this asset."
    )

    st.stop()


# ============================================================
# MULTI-INDEX FIX
# ============================================================

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)


# ============================================================
# INDICATORS
# ============================================================

data = calculate_indicators(data)


# ============================================================
# TARGET
# ============================================================

model_data = create_target(data)


# ============================================================
# ASSET HEADER
# ============================================================

st.html(f"""
<div style="margin-top:20px; margin-bottom:20px;">

    <div style="
        color:#64748b;
        font-size:13px;
        font-weight:700;
        letter-spacing:1px;
    ">
        BOJET MARKET ANALYTICS
    </div>

    <div style="
        color:#f8fafc;
        font-size:30px;
        font-weight:800;
    ">
        {asset_name}
    </div>

    <div style="
        color:#94a3b8;
        font-size:14px;
    ">
        {ticker_symbol}
    </div>

</div>
""")


# ============================================================
# LATEST DATA
# ============================================================

latest = data.iloc[-1]

current_price = float(
    latest["Close"]
)

if pd.notna(latest["MA10"]):

    ma10 = float(
        latest["MA10"]
    )

else:

    ma10 = current_price


if pd.notna(latest["RSI"]):

    rsi = float(
        latest["RSI"]
    )

else:

    rsi = 50


# ============================================================
# TREND
# ============================================================

if current_price > ma10:

    market_trend = "BULLISH"

else:

    market_trend = "BEARISH"


# ============================================================
# RSI
# ============================================================

if rsi > 70:

    rsi_status = "OVERBOUGHT"

elif rsi < 30:

    rsi_status = "OVERSOLD"

else:

    rsi_status = "NEUTRAL"


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.html("""
<div class="section-title">
    Market Overview
</div>
""")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            Current Price
        </div>

        <div class="metric-value">
            {currency_symbol}{current_price:,.2f}
        </div>

        <div class="metric-sub">
            {ticker_symbol}
        </div>

    </div>
    """)


with col2:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            10-Day Average
        </div>

        <div class="metric-value">
            {currency_symbol}{ma10:,.2f}
        </div>

        <div class="metric-sub">
            Moving Average
        </div>

    </div>
    """)


with col3:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            Market Trend
        </div>

        <div class="metric-value">
            {market_trend}
        </div>

        <div class="metric-sub">
            Price vs MA10
        </div>

    </div>
    """)


with col4:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            RSI
        </div>

        <div class="metric-value">
            {rsi:.2f}
        </div>

        <div class="metric-sub">
            {rsi_status}
        </div>

    </div>
    """)


# ============================================================
# MARKET CHART
# ============================================================

st.html("""
<div class="section-title">
    📈 Market Chart
</div>
""")


fig = go.Figure()


fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    )
)


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA10"],
        mode="lines",
        name="MA10"
    )
)


fig.update_layout(
    height=550,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)


st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# AI MODEL
# ============================================================

st.html("""
<div class="section-title">
    🤖 Bojet AI Forecast
</div>
""")


try:

    X, y = prepare_data(
        model_data
    )

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        accuracy
    ) = train_model(
        X,
        y
    )

    validation_scores, average_accuracy = (
        validate_model(
            X,
            y
        )
    )

    latest_features = (
        data.iloc[[-1]].copy()
    )

    ai_result = predict_latest(
        model,
        latest_features
    )

    ai_signal = ai_result["signal"]

    up_probability = (
        ai_result["up_probability"]
    )

    down_probability = (
        ai_result["down_probability"]
    )

    confidence = (
        ai_result["confidence"]
    )

except Exception as error:

    st.error(
        f"AI model could not be calculated: {error}"
    )

    st.stop()


# ============================================================
# AI CARD
# ============================================================

if ai_signal == "UP":

    ai_signal_class = "bullish"

else:

    ai_signal_class = "bearish"


st.html(f"""
<div class="ai-card">

    <div class="ai-title">
        Machine Learning Market Forecast
    </div>

    <div class="ai-signal {ai_signal_class}">
        {ai_signal}
    </div>

    <div class="ai-confidence">
        AI Confidence:
        <b>{confidence}</b>
    </div>

</div>
""")


# ============================================================
# PROBABILITIES
# ============================================================

prob1, prob2 = st.columns(2)


with prob1:

    st.html(f"""
    <div class="prob-card">

        <div class="prob-title">
            UP Probability
        </div>

        <div class="prob-value">
            {up_probability:.2f}%
        </div>

    </div>
    """)


with prob2:

    st.html(f"""
    <div class="prob-card">

        <div class="prob-title">
            DOWN Probability
        </div>

        <div class="prob-value">
            {down_probability:.2f}%
        </div>

    </div>
    """)


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

st.html("""
<div class="section-title">
    📊 Technical Analysis
</div>
""")


rsi_col, macd_col, bb_col = st.columns(3)


# ---------------- RSI ----------------

with rsi_col:

    st.markdown("### RSI")

    st.metric(
        "Value",
        f"{rsi:.2f}"
    )

    st.write(
        f"Status: **{rsi_status}**"
    )


# ---------------- MACD ----------------

macd = float(
    latest["MACD"]
)

signal_line = float(
    latest["Signal"]
)


if macd > signal_line:

    macd_status = "BULLISH"

else:

    macd_status = "BEARISH"


with macd_col:

    st.markdown("### MACD")

    st.metric(
        "MACD",
        f"{macd:.2f}"
    )

    st.write(
        f"Signal Line: **{signal_line:.2f}**"
    )

    st.write(
        f"Status: **{macd_status}**"
    )


# ---------------- BOLLINGER ----------------

bb_upper = float(
    latest["BB_Upper"]
)

bb_lower = float(
    latest["BB_Lower"]
)


if current_price > bb_upper:

    bb_status = "ABOVE UPPER BAND"

elif current_price < bb_lower:

    bb_status = "BELOW LOWER BAND"

else:

    bb_status = "WITHIN BANDS"


with bb_col:

    st.markdown("### Bollinger Bands")

    st.write(
        f"Upper: **{currency_symbol}{bb_upper:,.2f}**"
    )

    st.write(
        f"Lower: **{currency_symbol}{bb_lower:,.2f}**"
    )

    st.write(
        f"Status: **{bb_status}**"
    )


# ============================================================
# DECISION ENGINE
# ============================================================

st.html("""
<div class="section-title">
    🧠 Bojet Decision Engine
</div>
""")


# AI score

if confidence == "VERY HIGH":

    ai_score = 3

elif confidence == "HIGH":

    ai_score = 2

elif confidence == "MODERATE":

    ai_score = 1

elif confidence == "LOW":

    ai_score = 0.5

else:

    ai_score = 0


if ai_signal == "DOWN":

    ai_score *= -1


# Trend

if market_trend == "BULLISH":

    trend_score = 1

else:

    trend_score = -1


# RSI

if rsi < 30:

    rsi_score = 1

elif rsi > 70:

    rsi_score = -1

else:

    rsi_score = 0


# MACD

if macd > signal_line:

    macd_score = 1

else:

    macd_score = -1


# Bollinger

if current_price < bb_lower:

    bb_score = 1

elif current_price > bb_upper:

    bb_score = -1

else:

    bb_score = 0


# Final score

decision_score = (
    ai_score
    + trend_score
    + rsi_score
    + macd_score
    + bb_score
)


# Overall decision

if decision_score >= 3:

    overall_signal = "BULLISH"

    overall_class = "bullish"

elif decision_score <= -3:

    overall_signal = "BEARISH"

    overall_class = "bearish"

else:

    overall_signal = "NEUTRAL"

    overall_class = "neutral"


# ============================================================
# DECISION CARD
# ============================================================

st.html(f"""
<div class="decision-card">

    <div class="decision-label">
        BOJET SCORE
    </div>

    <div class="decision-score">
        {decision_score:.1f}
    </div>

    <div class="decision-signal {overall_class}">
        {overall_signal}
    </div>

</div>
""")


# ============================================================
# DECISION BREAKDOWN
# ============================================================

st.html("""
<div class="section-title">
    Decision Breakdown
</div>
""")


breakdown1, breakdown2, breakdown3 = st.columns(3)


with breakdown1:

    st.write(
        f"🤖 **AI Signal:** {ai_signal}"
    )

    st.write(
        f"🎯 **AI Confidence:** {confidence}"
    )


with breakdown2:

    st.write(
        f"📈 **Trend:** {market_trend}"
    )

    st.write(
        f"📊 **RSI:** {rsi_status}"
    )


with breakdown3:

    st.write(
        f"〽️ **MACD:** {macd_status}"
    )

    st.write(
        f"📉 **Bollinger:** {bb_status}"
    )


# ============================================================
# AI MODEL PERFORMANCE
# ============================================================

st.html("""
<div class="section-title">
    🎯 AI Model Performance
</div>
""")


perf1, perf2, perf3 = st.columns(3)


with perf1:

    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with perf2:

    st.metric(
        "Training Samples",
        len(X_train)
    )


with perf3:

    st.metric(
        "Testing Samples",
        len(X_test)
    )


st.info(
    "Model accuracy is based on historical test data "
    "and does not guarantee future market performance."
)


# ============================================================
# AI FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🧠 AI Feature Importance"
)


importance_df = get_feature_importance(
    model
).copy()


importance_df["Importance"] = (
    importance_df["Importance"] * 100
)


importance_df["Importance"] = (
    importance_df["Importance"].round(2)
)


st.dataframe(
    importance_df,
    column_config={
        "Feature": "AI Feature",
        "Importance": st.column_config.NumberColumn(
            "Importance (%)",
            format="%.2f%%"
        )
    },
    hide_index=True,
    width="stretch"
)


# ============================================================
# AI FEATURE SET EXPERIMENT
# ============================================================

st.subheader(
    "🧪 AI Feature Set Experiment"
)


try:

    feature_results = compare_feature_sets(
        data
    )

    feature_results["Test Accuracy"] = (
        feature_results["Test Accuracy"]
        .round(2)
    )

    st.dataframe(
        feature_results,
        column_config={
            "Feature Set": "Feature Set",

            "Features": st.column_config.NumberColumn(
                "Number of Features",
                format="%d"
            ),

            "Test Accuracy": st.column_config.NumberColumn(
                "Test Accuracy",
                format="%.2f%%"
            )
        },
        hide_index=True,
        width="stretch"
    )

except Exception as error:

    st.warning(
        f"Feature set experiment unavailable: {error}"
    )


# ============================================================
# MA10 BACKTEST
# ============================================================

st.html("""
<div class="section-title">
    📈 Strategy Backtesting
</div>
""")


st.html("""
<div class="info-box">
    Historical simulation of the MA10 trend-following strategy.
    Results are educational and do not guarantee future performance.
</div>
""")


try:

    backtest_results, backtest_summary = (
        backtest_strategy(
            data,
            initial_capital=10000
        )
    )

except Exception as error:

    st.error(
        f"MA10 backtest failed: {error}"
    )

    backtest_results = None

    backtest_summary = None


if backtest_summary is not None:

    bt1, bt2, bt3, bt4 = st.columns(4)


    with bt1:

        st.metric(
            "Initial Capital",
            f"${backtest_summary['initial_capital']:,.2f}"
        )


    with bt2:

        st.metric(
            "Final Value",
            f"${backtest_summary['final_value']:,.2f}"
        )


    with bt3:

        st.metric(
            "Return",
            f"{backtest_summary['total_return']:+.2f}%"
        )


    with bt4:

        st.metric(
            "Win Rate",
            f"{backtest_summary['win_rate']:.2f}%"
        )


    bt5, bt6 = st.columns(2)


    with bt5:

        st.metric(
            "Max Drawdown",
            f"{backtest_summary['max_drawdown']:.2f}%"
        )


    with bt6:

        completed_trades = (
            backtest_summary["winning_trades"]
            + backtest_summary["losing_trades"]
        )

        st.metric(
            "Completed Trades",
            completed_trades
        )


    # ========================================================
    # PORTFOLIO GROWTH
    # ========================================================

    st.markdown(
        "### 📊 Portfolio Growth"
    )


    portfolio_fig = go.Figure()


    portfolio_fig.add_trace(
        go.Scatter(
            x=backtest_results.index,
            y=backtest_results["Portfolio"],
            mode="lines",
            name="MA10 Strategy"
        )
    )


    portfolio_fig.update_layout(
        height=400,
        template="plotly_dark",
        yaxis_title="Portfolio Value",
        xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        portfolio_fig,
        width="stretch"
    )


    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    st.markdown(
        "### 📋 Trade Statistics"
    )


    st.write(
        f"**Winning Trades:** "
        f"{backtest_summary['winning_trades']}"
    )


    st.write(
        f"**Losing Trades:** "
        f"{backtest_summary['losing_trades']}"
    )


    st.write(
        f"**Total Completed Trades:** "
        f"{completed_trades}"
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown(
        "### 💰 Performance"
    )


    st.write(
        f"**Initial Capital:** "
        f"${backtest_summary['initial_capital']:,.2f}"
    )


    st.write(
        f"**Final Value:** "
        f"${backtest_summary['final_value']:,.2f}"
    )


    st.write(
        f"**Total Return:** "
        f"{backtest_summary['total_return']:+.2f}%"
    )


    st.write(
        f"**Maximum Drawdown:** "
        f"{backtest_summary['max_drawdown']:.2f}%"
    )


    # ========================================================
    # TRADE HISTORY
    # ========================================================

    st.markdown(
        "### 🔄 Trade History"
    )


    if not backtest_summary["trades"].empty:

        trade_history = (
            backtest_summary["trades"]
            .copy()
        )


        trade_history["Price"] = (
            trade_history["Price"]
            .map(
                lambda x:
                f"{currency_symbol}{x:,.2f}"
            )
        )


        trade_history["Shares"] = (
            trade_history["Shares"]
            .map(
                lambda x:
                f"{x:,.4f}"
            )
        )


        trade_history["Profit"] = (
            trade_history["Profit"]
            .map(
                lambda x:
                f"{currency_symbol}{x:,.2f}"
            )
        )


        st.dataframe(
            trade_history,
            width="stretch",
            height=400,
            hide_index=True
        )


# ============================================================
# STRATEGY COMPARISON
# ============================================================

st.html("""
<div class="section-title">
    📊 Strategy Comparison
</div>
""")


st.html("""
<div class="info-box">
    Comparison of MA10, Bojet AI and Buy & Hold using the
    same historical evaluation period.
</div>
""")


try:

    # --------------------------------------------------------
    # AI BACKTEST
    # --------------------------------------------------------

    ai_results, ai_summary = (
        backtest_ai_strategy(
            data,
            initial_capital=10000
        )
    )


    # --------------------------------------------------------
    # COMMON PERIOD
    # --------------------------------------------------------

    comparison_start = (
        ai_results.index[0]
    )


    comparison_data = (
        data.loc[
            data.index >= comparison_start
        ].copy()
    )


    # --------------------------------------------------------
    # MA10
    # --------------------------------------------------------

    (
        ma_results_compare,
        ma_summary_compare
    ) = backtest_strategy(
        comparison_data,
        initial_capital=10000
    )


    # --------------------------------------------------------
    # BUY AND HOLD
    # --------------------------------------------------------

    (
        buy_hold_results,
        buy_hold_summary
    ) = backtest_buy_and_hold(
        comparison_data,
        initial_capital=10000
    )


    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        {
            "Strategy": [
                "MA10 Strategy",
                "Bojet AI",
                "Buy & Hold"
            ],

            "Final Value": [
                ma_summary_compare["final_value"],
                ai_summary["final_value"],
                buy_hold_summary["final_value"]
            ],

            "Total Return": [
                ma_summary_compare["total_return"],
                ai_summary["total_return"],
                buy_hold_summary["total_return"]
            ],

            "Win Rate": [
                ma_summary_compare["win_rate"],
                ai_summary["win_rate"],
                None
            ],

            "Max Drawdown": [
                ma_summary_compare["max_drawdown"],
                ai_summary["max_drawdown"],
                buy_hold_summary["max_drawdown"]
            ]
        }
    )


    # --------------------------------------------------------
    # DISPLAY TABLE
    # --------------------------------------------------------

    comparison_display = (
        comparison_df.copy()
    )


    comparison_display["Final Value"] = (
        comparison_display["Final Value"]
        .map(
            lambda x:
            f"${x:,.2f}"
        )
    )


    comparison_display["Total Return"] = (
        comparison_display["Total Return"]
        .map(
            lambda x:
            f"{x:+.2f}%"
        )
    )


    comparison_display["Win Rate"] = (
        comparison_display["Win Rate"]
        .map(
            lambda x:
            "—"
            if pd.isna(x)
            else f"{x:.2f}%"
        )
    )


    comparison_display["Max Drawdown"] = (
        comparison_display["Max Drawdown"]
        .map(
            lambda x:
            f"{x:.2f}%"
        )
    )


    st.dataframe(
        comparison_display,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # STRATEGY PORTFOLIO CHART
    # ========================================================

    st.markdown(
        "### 📈 Strategy Portfolio Growth"
    )


    strategy_fig = go.Figure()


    strategy_fig.add_trace(
        go.Scatter(
            x=ma_results_compare.index,
            y=ma_results_compare["Portfolio"],
            mode="lines",
            name="MA10 Strategy"
        )
    )


    strategy_fig.add_trace(
        go.Scatter(
            x=ai_results.index,
            y=ai_results["Portfolio"],
            mode="lines",
            name="Bojet AI"
        )
    )


    strategy_fig.add_trace(
        go.Scatter(
            x=buy_hold_results.index,
            y=buy_hold_results["Portfolio"],
            mode="lines",
            name="Buy & Hold"
        )
    )


    strategy_fig.update_layout(
        height=500,
        template="plotly_dark",
        yaxis_title="Portfolio Value",
        xaxis_title="Date",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        strategy_fig,
        width="stretch"
    )


except Exception as error:

    st.warning(
        "Strategy comparison is unavailable for this period."
    )

    st.info(
        "Try selecting a longer historical period such as "
        "6mo, 1y or 2y."
    )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    <b>BOJET MARKET ANALYTICS</b>

    <br><br>

    AI • Technical Analysis • Market Intelligence

    <br><br>

    Educational project — not financial advice.

</div>
""")