import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Trader Performance vs Market Sentiment",
    layout="wide"
)

st.title("Trader Performance vs Market Sentiment")
st.markdown(
    """
    This dashboard explores how **market sentiment (Fear / Greed)**  
    influences **trader behavior, performance, and risk patterns**  
    using historical Hyperliquid trading data.
    """
)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("analysis_data.csv")

df = load_data()

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    with open("profitability_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_features.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

model, model_features = load_model()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters")

sentiment_choice = st.sidebar.selectbox(
    "Select Market Sentiment",
    sorted(df['sentiment'].dropna().unique())
)

activity_choice = st.sidebar.selectbox(
    "Trader Activity Group",
    ["All"] + sorted(df['activity_group'].dropna().unique().tolist())
)

filtered_df = df[df['sentiment'] == sentiment_choice]

if activity_choice != "All":
    filtered_df = filtered_df[filtered_df['activity_group'] == activity_choice]

# -------------------------
# Section 1 — Overview Metrics
# -------------------------
st.subheader("Overview Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average PnL", round(filtered_df['total_pnl'].mean(), 2))
col2.metric("Median PnL", round(filtered_df['total_pnl'].median(), 2))
col3.metric("Avg Trades / Day", round(filtered_df['num_trades'].mean(), 2))
col4.metric("Avg Trade Size (USD)", round(filtered_df['avg_trade_size'].mean(), 2))

# -------------------------
# Section 2 — PnL Distribution
# -------------------------
st.subheader("PnL Distribution")

fig, ax = plt.subplots()
sns.histplot(filtered_df['total_pnl'], bins=60, ax=ax)
ax.set_title(f"PnL Distribution — {sentiment_choice}")
ax.set_xlabel("Daily PnL")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# -------------------------
# Section 3 — Behavior Analysis
# -------------------------
st.subheader("Trader Behavior vs Sentiment")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Average Trades per Day**")
    trades_df = df.groupby('sentiment')['num_trades'].mean()
    st.bar_chart(trades_df)

with col2:
    st.markdown("**Average Trade Size (USD)**")
    size_df = df.groupby('sentiment')['avg_trade_size'].mean()
    st.bar_chart(size_df)

with col3:
    st.markdown("**Average Long/Short Ratio**")
    ls_df = df.groupby('sentiment')['long_short_ratio'].mean()
    st.bar_chart(ls_df)

# -------------------------
# Section 4 — Trader Segmentation
# -------------------------
st.subheader("Trader Segmentation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Activity Group Performance (Median PnL)**")
    activity_perf = (
        df.groupby(['activity_group', 'sentiment'])['total_pnl']
        .median()
        .unstack()
    )
    st.dataframe(activity_perf)

with col2:
    st.markdown("**Consistency Group Performance (Median PnL)**")
    consistency_perf = (
        df.groupby(['consistency_group', 'sentiment'])['total_pnl']
        .median()
        .unstack()
    )
    st.dataframe(consistency_perf)

# -------------------------
# Section 5 — Prediction Model
# -------------------------
st.subheader("Next-Day Profitability Prediction")

st.markdown(
    """
    This section uses a **lightweight predictive model** trained on  
    historical trader behavior and market sentiment to estimate the  
    probability of being profitable on the **next trading day**.
    """
)

# Prepare input features (average behavior under selected filters)
input_row = {
    'num_trades': filtered_df['num_trades'].mean(),
    'avg_trade_size': filtered_df['avg_trade_size'].mean(),
    'win_rate': filtered_df['win_rate'].mean(),
    'long_short_ratio': filtered_df['long_short_ratio'].mean(),
    'sentiment_Fear': 1 if sentiment_choice == 'Fear' else 0,
    'sentiment_Greed': 1 if sentiment_choice == 'Greed' else 0
}

input_df = pd.DataFrame([input_row])

# Align with model feature order
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Run prediction
if input_df.isnull().any().any():
    st.warning("Not enough data to generate prediction for this selection.")
else:
    prob = model.predict_proba(input_df)[0][1]
    prediction = "Profitable" if prob >= 0.5 else "Not Profitable"

    col1, col2 = st.columns(2)
    col1.metric("Predicted Outcome", prediction)
    col2.metric("Probability of Profit", f"{prob:.2%}")

# -------------------------
# Section 6 — Strategy Recommendations
# -------------------------
st.subheader("Strategy Recommendations")

st.markdown(
    """
    **Segment-aware trading during Fear**  
    - Fear periods increase volatility and opportunity for skilled traders.  
    - High-activity and consistent traders can participate actively.  
    - Low-activity traders should reduce exposure during Fear.

    **Risk control during Extreme Greed**  
    - Low-activity and inconsistent traders tend to underperform.  
    - Reducing trade frequency and exposure helps mitigate losses.
    """
)

st.markdown("---")
st.caption("Built as part of the Data Science Intern assignment — Primetrade.ai")
