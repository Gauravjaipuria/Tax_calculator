import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Page config
st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("\U0001F4C8 AI-Powered Stock Portfolio Optimizer")

# Sidebar selection for mode
st.sidebar.header("⚖️ Choose Analysis Mode")
mode = st.sidebar.radio("Forecast Mode", ["Stock", "Index"])

# Sidebar configuration for stock or index
st.sidebar.header("\U0001F527 Configuration")

country = st.sidebar.selectbox("Market", ["India", "America"]) if mode == "Stock" else None

# Mapping user-friendly index names to Yahoo Finance tickers
index_names = {
    "Sensex": "^BSESN",
    "Nifty 50": "^NSEI",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "FTSE 100": "^FTSE"
}

if mode == "Stock":
    stocks = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
    stock_list = [s.strip().upper() + ".NS" if country == "India" else s.strip().upper() for s in stocks.split(",")]
else:
    selected_index = st.sidebar.selectbox("Select Index", list(index_names.keys()))
    stock_list = [index_names[selected_index]]

years = st.sidebar.slider("Years of Historical Data", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)
if mode == "Stock":
    investment = st.sidebar.number_input("Total Investment (₹)", value=50000.0)
    risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])
    risk_map = {"Low": 1, "Medium": 2, "High": 3}
    risk_level = risk_map[risk_profile]

# Containers
forecasted_prices = {}
volatilities = {}
trend_signals = {}
actual_vs_predicted = {}
xgb_forecasts = {}
rf_forecasts = {}

# Loop through each stock/index
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

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    xgb_pred = xgb_model.predict(test[['Lag_1']])
    future_xgb = [xgb_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    rf_pred = rf_model.predict(test[['Lag_1']])
    future_rf = [rf_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    xgb_forecasts[stock] = xgb_pred[-1]
    rf_forecasts[stock] = rf_pred[-1]
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    actual_vs_predicted[stock] = (test['Close'], xgb_pred, rf_pred)

    # Plot forecast
    future_dates = pd.date_range(df.index[-1], periods=forecast_days+1, freq='B')[1:]
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

if mode == "Stock":
    # Allocation
    risky_stocks = [s for s in volatilities if volatilities[s] > 0.03]
    safe_stocks = [s for s in volatilities if s not in risky_stocks]
    risk_alloc_pct = {1: 0.7, 2: 0.5, 3: 0.3}[risk_level]
    risky_amt = investment * risk_alloc_pct
    safe_amt = investment - risky_amt
    allocation = {}
    for s in risky_stocks:
        allocation[s] = risky_amt / len(risky_stocks) if risky_stocks else 0
    for s in safe_stocks:
        allocation[s] = safe_amt / len(safe_stocks) if safe_stocks else 0
    total_alloc = sum(allocation.values())
    alloc_percent = {s: round((amt / total_alloc) * 100, 4) for s, amt in allocation.items()}

    st.subheader("\U0001F4B8 Optimized Portfolio Allocation")
    alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
    alloc_df['Allocation (%)'] = alloc_df.index.map(lambda s: alloc_percent[s])
    st.dataframe(alloc_df.style.format({'Investment Amount (₹)': '₹{:,.2f}', 'Allocation (%)': '{:.2f}%'}))

st.subheader("\U0001F4C8 AI Trend Signals")
trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
st.dataframe(trend_df)

st.subheader("\U0001F52E Forecasted Prices")
forecast_df = pd.DataFrame.from_dict(forecasted_prices, orient='index')
st.dataframe(forecast_df.style.format("{:.2f}"))

st.subheader("\U0001F4CA Risk Classification by Volatility")
risk_tiers = {
    s: "3 (High Risk)" if vol > 0.03 else "2 (Medium Risk)" if vol > 0.01 else "1 (Low Risk)"
    for s, vol in volatilities.items()
}
risk_df = pd.DataFrame.from_dict(risk_tiers, orient='index', columns=['Risk Level'])
st.dataframe(risk_df)

st.subheader("\U0001F4C9 Backtest: Actual vs Predicted")
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

st.subheader("\U0001F4CC Sharpe Ratio & Annual Return")
sharpe_rows = []
risk_free_rate = 0.05
for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    df['Returns'] = df['Close'].pct_change()
    annual_return = df['Returns'].mean() * 252
    annual_volatility = df['Returns'].std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    sharpe_rows.append([
        stock,
        round(annual_return, 4),
        round(annual_volatility, 4),
        round(sharpe, 4)
    ])
sharpe_df = pd.DataFrame(sharpe_rows, columns=['Stock', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio'])
st.dataframe(sharpe_df.set_index('Stock').style.format({
    'Annual Return': '{:.2%}',
    'Annual Volatility': '{:.2%}',
    'Sharpe Ratio': '{:.2f}'
}))
