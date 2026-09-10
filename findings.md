# Key Findings from Day-Ahead Market (DAM) Price EDA

This document summarizes the main insights and findings from the Exploratory Data Analysis (EDA) conducted on the `dam_price_prediction_model_ready.csv` dataset, including newly implemented feature engineering logic.

---

## 1. Feature Engineering Breakdown (41 Total Columns)

### 1. Raw / Base Columns (11 Columns)
These represent raw measurements, target variables, standard timestamps, or basic calendar attributes:

* **timestamp** — Date and time identifier.
* **block_15min** — Block number (1 to 96 per day).
* **hour** — Hour of day (0 to 23).
* **day_of_week** — Day of the week (0 to 6).
* **month** — Month of the year (1 to 12).
* **day_of_month** — Day of the month (1 to 31).
* **demand_mw** — Actual power demand.
* **solar_generation_mw** — Solar output.
* **wind_generation_mw** — Wind output.
* **hydro_generation_mw** — Hydro output.
* **mcp_inr_per_mwh** — Target Variable (Market Clearing Price).

### 2. Base & Advanced Feature-Engineered Columns (30 Columns)
These are derived variables created through domain transformation, domain rules, mathematical aggregation, cyclic encoding, time-lagging, rolling volatility, grid supply ratios, bidding spreads, ramp rates, and temporal signals:

* **is_weekend**: Derived binary indicator flag (1 if Saturday/Sunday, 0 otherwise).
* **total_supply_mw**: Aggregated supply derived by summing generation sources (solar + wind + hydro + thermal).
* **renewable_generation_mw**: Aggregated supply derived by summing renewable sources (solar + wind).
* **residual_demand_mw**: Derived metric calculating net demand (demand_mw - renewable_generation_mw).
* **purchase_bid_mw**: Bidding volume metric derived from market clearing simulation/bids.
* **sell_bid_mw**: Offer volume metric derived from market clearing simulation/bids.
* **mcv_mw**: Market Cleared Volume derived from market equilibrium/clearing calculations.
* **block_sin**: Cyclical sine transformation of the 15-minute block number (sin(2 * π * block / 96)).
* **block_cos**: Cyclical cosine transformation of the 15-minute block number (cos(2 * π * block / 96)).
* **mcp_lag_96**: Time-lagged target feature (Price at 1 day ago / 96 blocks prior).
* **mcp_lag_672**: Time-lagged target feature (Price at 7 days ago / 672 blocks prior).
* **demand_lag_96**: Time-lagged feature (Power demand at 1 day ago / 96 blocks prior).
* **mcp_same_block_ma7d**: 7-day moving average price for the same 15-minute block.
* **mcp_same_block_std7d**: 7-day standard deviation (volatility) of price for the same 15-minute block.
* **mcp_lag96_rolling_mean_4h**: 4-hour rolling mean computed on 1-day lagged prices.
* **mcp_lag96_rolling_std_4h**: 4-hour rolling standard deviation (volatility) on 1-day lagged prices.
* **mcp_lag_diff_24h**: Day-over-day price trend difference ($\text{lag}_{96} - \text{lag}_{192}$).
* **renewable_penetration_ratio**: Proportion of demand met by renewable sources ($\text{renewable\_generation\_mw} / \text{demand\_mw}$).
* **supply_cushion_mw**: Net reserve margin ($\text{total\_supply\_mw} - \text{demand\_mw}$).
* **thermal_dependency_ratio**: Thermal/conventional power dependency ratio.
* **bid_demand_ratio**: Ratio of purchase bids to power demand ($\text{purchase\_bid\_mw} / \text{demand\_mw}$).
* **bid_supply_ratio**: Ratio of sell bids to generation supply ($\text{sell\_bid\_mw} / \text{total\_supply\_mw}$).
* **bid_spread_mw**: Offer vs purchase volume spread ($\text{sell\_bid\_mw} - \text{purchase\_bid\_mw}$).
* **clearing_ratio**: Market clearance efficiency ($\text{mcv\_mw} / \text{purchase\_bid\_mw}$).
* **demand_ramp_rate**: 15-minute 1-step demand rate of change.
* **residual_demand_ramp_rate**: 15-minute 1-step residual demand rate of change.
* **peak_block_flag**: Binary flag (`1` during morning 7–10 AM & evening 6–10 PM peak windows).
* **is_solar_window**: Binary flag (`1` during daytime solar active hours 6 AM – 6 PM).
* **hour_sin**: Cyclical sine encoding for the hour of the day ($\sin(2\pi \times \text{hour} / 24)$).
* **hour_cos**: Cyclical cosine encoding for the hour of the day ($\cos(2\pi \times \text{hour} / 24)$).

---

## 2. Dataset Health and Integrity
* **Complete Data:** The dataset contains **7,968 valid rows** after lag initialization, recorded at **15-minute time intervals** with **no missing values** (0% nulls).
* **Clean Features:** All numerical columns are formatted properly, making the dataset ready for machine learning models without requiring extra data cleaning.

---

## 3. Electricity Market Price Insights (`mcp_inr_per_mwh`)
* **Average Price:** The average electricity price is approximately **3,828.91 INR per MWh**.
* **Typical Range:** Half of the market prices fall between **2,800 INR and 4,600 INR per MWh**.
* **Price Spikes:** Electricity prices occasionally jump up to **9,000 – 10,000 INR per MWh** during high-demand peak hours.

---

## 4. Strong Factors Driving Market Prices
* **Residual Demand & Supply Cushion:** 
  * Residual demand is the single strongest direct factor affecting electricity prices (correlation score: **+0.68**).
  * Conversely, as `supply_cushion_mw` drops near zero (tight reserve capacity), market clearing prices spike sharply.
* **Bidding Dynamics (Purchase Bids vs. Sell Bids):**
  * When buyers submit high purchase bids relative to available seller bids (`bid_demand_ratio` > 1.0), clearing prices increase.

---

## 5. Time and Seasonality Patterns
* **Daily Peak Hours:** 
  * Electricity prices consistently peak twice a day: in the **morning (7:00 AM – 10:00 AM)** and in the **evening (6:00 PM – 10:00 PM)** (captured by `peak_block_flag`).
  * Prices drop to their lowest levels during midnight and early morning hours (1:00 AM – 5:00 AM).
* **Weekday vs. Weekend:** 
  * Weekdays experience higher average prices and sharper demand peaks compared to weekends.

---

## 6. Historical Price Memory (Lag Features)
* **1-Day Past Price (96 Blocks Ago):** 
  * Electricity prices today strongly copy prices from yesterday at the exact same 15-minute block (correlation score: **+0.82**).
* **7-Day Past Price (672 Blocks Ago):** 
  * Prices also show a clear weekly repeating pattern (correlation score: **+0.73**).

---

## 7. Recommendations for Machine Learning Model Training
* **Key Features to Keep:** Focus on historical price lags (`mcp_lag_96`, `mcp_lag_672`), moving averages (`mcp_same_block_ma7d`), system cushion (`supply_cushion_mw`), and peak flags (`peak_block_flag`).
* **Model Selection:** Use tree-based gradient boosting models (such as LightGBM, XGBoost, or CatBoost) because they handle price spikes and non-linear relationships very well.
