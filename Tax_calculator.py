import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# Sidebar inputs
st.sidebar.header("🔧 Configuration")
mode = st.sidebar.selectbox("Select Asset Type", ["Stock", "Index"])
years = st.sidebar.slider("Historical Data (Years)", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 90)
investment = st.sidebar.number_input("💰 Investment Amount (INR)", value=100000.0, step=1000.0)
risk_level = st.sidebar.selectbox("🎯 Your Risk Appetite", [1, 2, 3], format_func=lambda x: ["Low", "Medium", "High"][x - 1])

# Ticker input
if mode == "Stock":
    stocks_input = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
    market = st.sidebar.selectbox("Select Market", ["India", "America"])
    stock_list = [f"{symbol.strip().upper()}.NS" if market == "India" else symbol.strip().upper() for symbol in stocks_input.split(",")]
else:
    index_map = {
        "Sensex": "^BSESN",
        "Nifty 50": "^NSEI",
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "FTSE 100": "^FTSE"
    }
    index_name = st.sidebar.selectbox("Select Index", list(index_map.keys()))
    stock_list = [index_map[index_name]]

# Containers
forecasted_prices = {}
trend_signals = {}
xgb_forecasts = {}
rf_forecasts = {}
actual_vs_predicted = {}
sharpe_ratios = {}
returns = {}
fair_values = {}
expected_price_growths = {}
risk_levels = {}
volatilities = {}

RISK_FREE_RATE = 0.04

for stock in stock_list:
    st.markdown(f"## Analysis for: {'Index' if mode == 'Index' else 'Stock'} - `{stock}`")

    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.warning(f"⚠️ No data available for {stock}. Skipping.")
        continue

    df = df[['Close']].dropna()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    trend_signals[stock] = "🟢 Bullish (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "🔴 Bearish (Sell)"

    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    xgb_pred = xgb_model.predict(test[['Lag_1']])
    future_xgb = [xgb_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    rf_pred = rf_model.predict(test[['Lag_1']])
    future_rf = [rf_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    # Risk metrics
    daily_returns = df['Close'].pct_change().dropna()
    avg_return = np.mean(daily_returns) * 252
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (avg_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    volatility_level = "Low" if volatility < 0.15 else "Medium" if volatility < 0.3 else "High"

    returns[stock] = avg_return
    volatilities[stock] = volatility
    sharpe_ratios[stock] = sharpe_ratio
    risk_levels[stock] = volatility_level

    if mode == "Stock":
        last_eps = df['Close'].iloc[-1] / 15
        projected_eps = last_eps * (1 + avg_return)
        fair_values[stock] = projected_eps * 15
    else:
        fair_values[stock] = np.nan

    latest_price = df['Close'].iloc[-1]
    xgb_forecast_price = future_xgb[-1]
    expected_growth = ((xgb_forecast_price - latest_price) / latest_price) * 100
    expected_price_growths[stock] = expected_growth

    xgb_forecasts[stock] = xgb_pred[-1]
    rf_forecasts[stock] = rf_pred[-1]
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    actual_vs_predicted[stock] = (test['Close'], xgb_pred, rf_pred)

    # Plot
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label="Historical", color='black')
    plt.plot(df['MA_50'], label="50-Day MA", linestyle='--', color='blue')
    plt.plot(df['MA_200'], label="200-Day MA", linestyle='--', color='purple')
    plt.plot(future_dates, future_xgb, label="XGBoost Forecast", linestyle='--', color='red')
    plt.plot(future_dates, future_rf, label="RF Forecast", linestyle='--', color='green')
    plt.title(f"{stock} - Forecasted Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# Outputs
st.subheader("🔮 Forecasted Prices")
st.dataframe(pd.DataFrame.from_dict(forecasted_prices, orient='index').style.format("{:.2f}"))

st.subheader("📈 AI Trend Signals")
st.dataframe(pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal']))

st.subheader("📊 Risk & Return Metrics")
rr_df = pd.DataFrame({
    "Expected Annual Return": returns,
    "Volatility": volatilities,
    "Risk Level": risk_levels,
    "Sharpe Ratio": sharpe_ratios
})
st.dataframe(rr_df.style.format("{:.2f}"))

st.subheader("💸 Fair Value Estimate")
fair_df = pd.DataFrame.from_dict(fair_values, orient='index', columns=["Estimated Fair Value"])
st.dataframe(fair_df.style.format("{:.2f}"))

st.subheader("📈 Forecasted Price Growth (%)")
growth_df = pd.DataFrame.from_dict(expected_price_growths, orient='index', columns=["Forecasted Growth % (XGBoost)"])
st.dataframe(growth_df.style.format("{:.2f}"))

# Risk classification
low_risk = [s for s, v in volatilities.items() if v <= 0.01]
medium_risk = [s for s, v in volatilities.items() if 0.01 < v <= 0.03]
high_risk = [s for s, v in volatilities.items() if v > 0.03]

st.subheader("🔍 Stock Risk Classification")
risk_classification_df = pd.DataFrame({
    "Stock": low_risk + medium_risk + high_risk,
    "Risk Category": (["Low"] * len(low_risk)) + (["Medium"] * len(medium_risk)) + (["High"] * len(high_risk))
})
st.dataframe(risk_classification_df)

# Allocation
st.subheader("💸 Portfolio Allocation Based on Risk")
allocation = {}

if len(low_risk) == len(volatilities):
    per_stock = investment / len(low_risk)
    for stock in low_risk:
        allocation[stock] = per_stock
elif len(medium_risk) == len(volatilities):
    per_stock = investment / len(medium_risk)
    for stock in medium_risk:
        allocation[stock] = per_stock
elif len(high_risk) == len(volatilities):
    per_stock = investment / len(high_risk)
    for stock in high_risk:
        allocation[stock] = per_stock
else:
    risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}
    risky_allocation = investment * risk_allocation[risk_level]
    safe_allocation = investment - risky_allocation

    safe_stocks = low_risk + medium_risk if risk_level == 3 else low_risk
    risky_stocks = high_risk if risk_level == 3 else medium_risk + high_risk

    if risky_stocks:
        per_risky_stock = risky_allocation / len(risky_stocks)
        for stock in risky_stocks:
            allocation[stock] = per_risky_stock

    if safe_stocks:
        per_safe_stock = safe_allocation / len(safe_stocks)
        for stock in safe_stocks:
            allocation[stock] = per_safe_stock

# Display Allocation
total_allocation = sum(allocation.values())
alloc_percent = {stock: round((amount / total_allocation) * 100, 2) for stock, amount in allocation.items()}

alloc_df = pd.DataFrame({
    "Stock": allocation.keys(),
    "INR Allocation": allocation.values(),
    "Allocation %": alloc_percent.values()
})
st.dataframe(alloc_df.style.format({"INR Allocation": "₹{:.2f}", "Allocation %": "{:.2f}%"}))
