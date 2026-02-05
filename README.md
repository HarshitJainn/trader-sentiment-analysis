# Trader Performance vs Market Sentiment (Fear & Greed)

## Objective
The goal of this project is to analyze how **Bitcoin market sentiment (Fear / Greed)** relates to **trader behavior and performance** on Hyperliquid. The analysis aims to uncover behavioral patterns that can inform **smarter, sentiment-aware trading strategies**.

---

## Datasets
1. **Bitcoin Market Sentiment (Fear/Greed)**
   - Daily sentiment labels (Fear, Greed, Extreme Greed, Neutral)
   - Source: Fear & Greed Index
https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing


2. **Historical Trader Data (Hyperliquid)**
   - Trade-level data including account, trade size, direction, timestamps, and realized PnL
https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing


---


## Outputs
All analysis outputs (charts and tables) are rendered directly in the notebook (`analysis.ipynb`) and can be reproduced by running the notebook end-to-end.  
An optional Streamlit dashboard (`app.py`) is also provided for interactive exploration.



## Methodology

### Data Preparation
- Loaded and audited both datasets (rows, columns, missing values, duplicates)
- Converted timestamps and aligned both datasets at a **daily level**
- Aggregated trade-level data into **trader–day metrics**

### Key Metrics Engineered
- Daily total PnL per trader
- Win rate
- Number of trades per day
- Average trade size (USD)
- Long/Short ratio (directional bias)

> **Note:** Explicit leverage was not available; trade size and exposure were used as proxies for risk-taking behavior.

---

## Analysis Summary

### Performance vs Sentiment
- Trader performance differs across sentiment regimes
- Fear days are associated with higher volatility but not uniformly worse outcomes
- Median PnL provides a more reliable signal than mean PnL due to outliers

### Behavioral Changes
- Trade frequency and trade size increase during Greed and Extreme Greed
- Directional bias shifts with sentiment, indicating momentum-driven behavior

### Trader Segmentation
Traders were segmented into:
- **Frequent vs Infrequent traders**
- **Consistent vs Inconsistent traders**

Segment-level analysis shows that trader skill and engagement strongly moderate the impact of market sentiment.

---

## Key Insights
1. Market sentiment does not affect all traders equally; active and consistent traders perform better across regimes
2. Fear periods create opportunity for skilled traders due to increased volatility
3. Extreme Greed amplifies losses for low-activity and inconsistent traders

---

## Strategy Recommendations
1. **Fear Days**: Allow higher participation only for active and consistent traders; reduce exposure for low-activity traders
2. **Extreme Greed**: Limit exposure and trade frequency for weaker traders to avoid sentiment-driven losses

---

## Bonus Work
- Built a **simple predictive model** to estimate next-day trader profitability using sentiment and behavior features
- Clustered traders into behavioral archetypes
- Developed a **Streamlit dashboard** to explore performance, behavior, and predictive signals interactively

---

## How to Run

### Notebook
```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

### Streamlit Dashboard (Optional)
```bash
streamlit run app.py
```

---

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

---

## Repository Structure
```
trader-sentiment-analysis/
│
├── analysis.ipynb              # Main notebook (Part A, B, C)
├── README.md                   # Full explanation + summary
│
├── analysis_data.csv           # Processed data for dashboard
├── profitability_model.pkl     # Trained predictive model
├── model_features.pkl          # Feature order for inference
│
├── app.py                      # Streamlit dashboard (Bonus)
├── requirements.txt            # Dependencies
```

---

## Author
Harshit Jain
