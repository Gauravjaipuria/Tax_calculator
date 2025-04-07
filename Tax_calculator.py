import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# Page config
st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# Sidebar
st.sidebar.header("🔧 Configuration")
mode = st.sidebar.selectbox("Select Type", ["Stock", "Index"])
years = st.sidebar.slider("Years of Historical Data", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)

if mode == "Stock":
    stocks = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
    country = st.sidebar.selectbox("Market", ["India", "America"])
    stock_list = [s.strip().upper() + ".NS" if country == "India" else s.strip().upper() for s in stocks.split(",")]
elif mode == "Index":
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

# Data containers
forecasted_prices = {}
trend_signals = {}
xgb_forecasts = {}
rf_forecasts = {}
actual_vs_predicted = {}
sharpe_ratios = {}
returns = {}
fair_values = {}
expected_price_growths = {}
risk_free_rate = 0.04  # Assumed constant for Sharpe Ratio calculation

# Forecasting loop
for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.warning(f"⚠️ No data for {stock}. Skipping.")
        continue

    df = df[['Close']].dropna()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    trend_signals[stock] = "🟢 Bullish (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "🔴 Bearish (Sell)"

    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # XGBoost
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    xgb_pred = xgb_model.predict(test[['Lag_1']])
    future_xgb = [xgb_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    rf_pred = rf_model.predict(test[['Lag_1']])
    future_rf = [rf_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    # Returns & Sharpe Ratio
    daily_returns = df['Close'].pct_change().dropna()
    avg_return = np.mean(daily_returns) * 252
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
    sharpe_ratios[stock] = sharpe_ratio
    returns[stock] = avg_return

    # Fair Value Estimation using Forward PE (assume PE = 15)
    last_eps = df['Close'].iloc[-1] / 15
    projected_eps = last_eps * (1 + avg_return)
    fair_value = projected_eps * 15
    fair_values[stock] = fair_value

    # Forecasted Growth
    latest_price = df['Close'].iloc[-1]
    xgb_forecast_price = future_xgb[-1]
    expected_growth = ((xgb_forecast_price - latest_price) / latest_price) * 100
    expected_price_growths[stock] = expected_growth

    # Store outputs
    xgb_forecasts[stock] = xgb_pred[-1]
    rf_forecasts[stock] = rf_pred[-1]
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    actual_vs_predicted[stock] = (test['Close'], xgb_pred, rf_pred)

    # Forecast Chart
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label="Historical", color='black')
    plt.plot(df['MA_50'], label="50-Day MA", linestyle='--', color='blue')
    plt.plot(df['MA_200'], label="200-Day MA", linestyle='--', color='purple')
    plt.plot(future_dates, future_xgb, label="XGBoost Forecast", linestyle='--', color='red')
    plt.plot(future_dates, future_rf, label="RF Forecast", linestyle='--', color='green')
    plt.title(f"{stock} - Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# ==================== Display Section ==================== #
st.subheader("🔮 Forecasted Prices")
forecast_df = pd.DataFrame.from_dict(forecasted_prices, orient='index')
st.dataframe(forecast_df.style.format("{:.2f}"))

st.subheader("📈 AI Trend Signals")
trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
st.dataframe(trend_df)

st.subheader("📉 Backtest: Actual vs Predicted")
for stock in actual_vs_predicted:
    actual, xgb_pred, rf_pred = actual_vs_predicted[stock]
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual, label='Actual', color='black')
    plt.plot(actual.index, xgb_pred, label='XGBoost', color='red')
    plt.plot(actual.index, rf_pred, label='Random Forest', color='green')
    plt.title(f"{stock} - Backtest Performance")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

st.subheader("📊 Risk & Return Metrics")
rr_df = pd.DataFrame({
    "Expected Annual Return": returns,
    "Sharpe Ratio": sharpe_ratios
})
st.dataframe(rr_df.style.format("{:.2f}"))

st.subheader("💸 Fair Value Estimate")
fair_df = pd.DataFrame.from_dict(fair_values, orient='index', columns=["Estimated Fair Value"])
st.dataframe(fair_df.style.format("{:.2f}"))

st.subheader("📈 Forecasted Price Growth (%)")
growth_df = pd.DataFrame.from_dict(expected_price_growths, orient='index', columns=["Forecasted Growth % (XGBoost)"])
st.dataframe(growth_df.style.format("{:.2f}"))
