# ==========================
# Step 1: Setup and Libraries
# ==========================

# Streamlit for web app UI
import streamlit as st

# For fetching stock data
import yfinance as yf

# Data manipulation
import pandas as pd
import numpy as np

# Charts
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# ==========================
# Info for beginners
# ==========================
st.set_page_config(
    page_title="Stock Price Predictor App",
    layout="wide"
)

st.title("📈 Stock Price Prediction App")
st.markdown("""
This app allows you to:
- Fetch historical stock data.
- Visualize prices and volumes.
- Predict future stock prices using a simple Linear Regression model.
- Download historical and predicted data.

All designed for **beginners** to learn and experiment.
""")

st.sidebar.header("🔧 Select Options")

# Info for Indian stocks
st.sidebar.info(
    "💡 For Indian stocks listed on NSE, please add '.NS' at the end of the ticker.\n"
    "Example: Reliance Industries → RELIANCE.NS, Tata Motors → TATAMOTORS.NS"
)

# 1️⃣ Mapping of top 10 popular US companies
company_tickers = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "Meta": "META",
    "NVIDIA": "NVDA",
    "Netflix": "NFLX",
    "Disney": "DIS",
    "Intel": "INTC"
}

# 2️⃣ Stock Ticker / Company Input
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker or Company Name (e.g., AAPL or Apple):", 
    value="AAPL"
)

# Convert company name to ticker if found, else use uppercase input
ticker = company_tickers.get(ticker_input.title(), ticker_input.upper())

# 3️⃣ Validate the ticker using yfinance
try:
    test_data = yf.Ticker(ticker).history(period="1d")
    if test_data.empty:
        st.error(f"❌ '{ticker_input}' is not a valid ticker. Please try a correct ticker or company name.")
        st.stop()  # Stop execution if invalid
except Exception as e:
    st.error(f"❌ Error fetching data for '{ticker_input}': {e}")
    st.stop()

# 4️⃣ Date Picker for Start and End Dates
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# 5️⃣ Prediction Button
predict_button = st.sidebar.button("Predict Future Price")

# 6️⃣ Optional: Number of days to predict
future_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=5)

# --------- Main Panel ---------
st.subheader(f"📊 Selected Stock: {ticker} ({ticker_input.title()})")
st.write(f"Data from **{start_date}** to **{end_date}**")

# ==========================
# Step 3: Historical Data
# ==========================

st.header("📖 Introduction to Stock Prices (OHLCV)")

st.markdown("""
**OHLCV Explained:**
- **Open (O):** The price at which the stock started trading that day.
- **High (H):** The highest price reached during the trading day.
- **Low (L):** The lowest price reached during the trading day.
- **Close (C):** The final price at the end of the trading day (**our target for prediction**).
- **Volume (V):** Total number of shares traded during the day.

This app uses **Open, High, Low, and Volume** as features to predict **Close price** for the selected stock.
""")

# --------- Load Historical Data ---------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    
    # Flatten columns if multi-index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data

# Load the data
data = load_data(ticker, start_date, end_date)

# Check if data is empty
if data.empty:
    st.error(f"❌ No data found for {ticker} between {start_date} and {end_date}.")
    st.stop()

# --------- Prepare data for display ---------

# Reset index so Date becomes a column
data_display = data.reset_index()

# Format Date column to show only YYYY-MM-DD (remove time)
data_display['Date'] = data_display['Date'].dt.strftime('%Y-%m-%d')

# Set Date as the index to remove extra serial number column
data_display = data_display.set_index('Date')

# Display the table
st.subheader("📊 Historical Stock Data")
st.dataframe(data_display)

# Simple interactive line chart
fig_close = px.line(
    data, 
    x=data.index, 
    y='Close', 
    title=f"{ticker} Closing Price",
    labels={'Close': 'Closing Price', 'index': 'Date'},
    template='plotly_white'  # Clean and easy to read
)

# Add hover info
fig_close.update_traces(
    hovertemplate="Date: %{x}<br>Close Price: %{y:.2f}"
)

# Center the title
fig_close.update_layout(title={'x':0.5})

# Display in Streamlit
st.subheader("📈 Closing Price Chart")
st.plotly_chart(fig_close, use_container_width=True)

# --------- Optional: Volume Bar Chart ---------
st.subheader("📊 Volume Chart")
fig_volume = px.bar(data, x=data.index, y="Volume", title=f"{ticker} Volume Traded")
st.plotly_chart(fig_volume, use_container_width=True)

# ==========================
# Step 4: Predictive Model with Train/Test Split
# ==========================
st.header("🤖 Stock Price Prediction")

# -------------------------
# Select number of days to predict
# -------------------------
future_days = st.slider("Select number of days to predict into the future:", min_value=1, max_value=30, value=5)

# -------------------------
# Prepare features and target
# -------------------------
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features]
y = data[target]

# -------------------------
# Train/Test Split (80/20)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------------
# Train Linear Regression Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Model Performance
# -------------------------
# Training performance
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Testing performance
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

st.subheader("📊 Model Performance")
st.write("**Training Data:**")
st.write(f"MAE: **{train_mae:.2f}**, RMSE: **{train_rmse:.2f}**, R²: **{train_r2:.2f}**")
st.write("**Testing Data:**")
st.write(f"MAE: **{test_mae:.2f}**, RMSE: **{test_rmse:.2f}**, R²: **{test_r2:.2f}**")

# -------------------------
# Predict next N days (naive approach)
# -------------------------
last_row = X.tail(1)
future_predictions = []
last_features = last_row.values.flatten()

for i in range(future_days):
    pred = model.predict([last_features])[0]
    future_predictions.append(pred)
    # Update features for next prediction
    last_features[0] = pred  # Open
    last_features[1] = pred  # High
    last_features[2] = pred  # Low
    # Volume remains the same

# -------------------------
# Prepare future dates
# -------------------------
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')
future_dates_str = future_dates.strftime('%Y-%m-%d')

# -------------------------
# Show predictions in a table
# -------------------------
pred_df = pd.DataFrame({'Date': future_dates_str, 'Predicted Close': future_predictions})
pred_df = pred_df.set_index('Date')
st.subheader(f"📋 Predicted Close Prices for Next {future_days} Days")
st.dataframe(pred_df)

# -------------------------
# Overlay predictions on historical chart
# -------------------------
fig_pred = px.line(data, x=data.index, y='Close', title=f"{ticker} Closing Price with Predictions", labels={'Close':'Close Price', 'index':'Date'})
fig_pred.add_scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Predicted Close', line=dict(color='orange'))
fig_pred.update_layout(title={'x':0.5})
st.subheader("📈 Closing Price Chart with Predictions")
st.plotly_chart(fig_pred, use_container_width=True)

# -------------------------
# Educational Notes
# -------------------------
with st.expander("💡 How Predictions Are Made"):
    st.write("""
    - We split historical data into **training (80%)** and **testing (20%)** sets to evaluate model performance.
    - Linear Regression predicts Close prices using Open, High, Low, and Volume.
    - Metrics (MAE, RMSE, R²) are calculated separately for training and testing sets.
    - Predictions for future days use the last known features as a naive approach.
    - Keep in mind: future predictions are **uncertain** and may differ from actual stock prices.
    """)

# ==========================
# Step 5: Optional Enhancements
# ==========================
st.header("📈 Technical Indicators & Daily Returns")

# -------------------------
# Calculate SMA and EMA
# -------------------------
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

# -------------------------
# Calculate Daily Returns (%)
# -------------------------
data['Daily Return %'] = data['Close'].pct_change() * 100

# -------------------------
# Plot Closing Price with SMA & EMA
# -------------------------
fig_sma = px.line(data, x=data.index, y=['Close', 'SMA_10', 'EMA_10'],
                  title=f"{ticker} Closing Price with SMA & EMA",
                  labels={'value':'Price', 'index':'Date', 'variable':'Legend'})
st.plotly_chart(fig_sma, use_container_width=True)

# -------------------------
# Plot Daily Returns
# -------------------------
fig_return = px.bar(data, x=data.index, y='Daily Return %',
                    title=f"{ticker} Daily Returns (%)",
                    labels={'Daily Return %':'% Change', 'index':'Date'})
st.plotly_chart(fig_return, use_container_width=True)

# -------------------------
# Educational Notes
# -------------------------
with st.expander("💡 About These Indicators"):
    st.write("""
    - **SMA (Simple Moving Average):** Average Close price over last N days. Smooths out short-term fluctuations.
    - **EMA (Exponential Moving Average):** Similar to SMA but gives more weight to recent prices. Reacts faster to changes.
    - **Daily Returns:** Percentage change in Close price from previous day. Shows volatility.
    - **How to use:** 
        - SMA & EMA help identify trends: if price > SMA/EMA → bullish, if price < SMA/EMA → bearish.
        - RSI (not included yet) would indicate overbought/oversold conditions.
    - These indicators are **common in real-world trading** and help beginners understand stock behavior.
    """)

# ==========================
# Step 6: Download Options & Beginner Guidance
# ==========================
st.header("💾 Download Data & Guidance")

# -------------------------
# Download Historical Data
# -------------------------
st.subheader("Download Historical Data")
csv_hist = data.copy()
csv_hist.index.name = 'Date'
csv_hist = csv_hist.reset_index()  # convert index to column for CSV

st.download_button(
    label="📥 Download Historical Data as CSV",
    data=csv_hist.to_csv(index=False),
    file_name=f"{ticker}_historical_data.csv",
    mime='text/csv'
)

# -------------------------
# Download Predicted Data
# -------------------------
st.subheader("Download Predicted Data")
pred_download = pred_df.copy()
pred_download = pred_download.reset_index()  # make Date a column

st.download_button(
    label="📥 Download Predicted Close Prices as CSV",
    data=pred_download.to_csv(index=False),
    file_name=f"{ticker}_predicted_close.csv",
    mime='text/csv'
)

# -------------------------
# Beginner-Friendly Guidance
# -------------------------
st.subheader("📝 How to Use This App")

with st.expander("📌 Tips & Guidance for Beginners"):
    st.write("""
    1. **Charts:**  
       - The line charts show **historical closing prices**, SMA, EMA, and predicted future prices.  
       - Hover over points to see exact values.

    2. **Predicted Prices:**  
       - Predictions are based on a simple Linear Regression model.  
       - They **give an estimate** of future prices; real stock prices can vary.  

    3. **Daily Returns:**  
       - Shows percentage change in closing price day-to-day.  
       - Negative values → price dropped, Positive → price rose.  

    4. **Technical Indicators:**  
       - SMA and EMA help identify trends: if price > SMA/EMA → bullish, if price < SMA/EMA → bearish.  

    5. **Downloads:**  
       - You can download both historical and predicted data as CSV files for further analysis or practice.  

    6. **Learning Tip:**  
       - Experiment with different stocks, date ranges, and number of future days to **see how predictions change**.  
       - Try comparing predicted values to actual stock performance for learning purposes.
    """)

# -------------------------
# Tooltips for Inputs
# -------------------------
st.subheader("💡 Input Guidance")
st.info("""
- **Stock Ticker:** Enter a valid stock symbol (e.g., AAPL, TSLA, INFY.NS)  
- **Date Range:** Select start and end dates for historical data  
- **Number of Future Days:** Choose how many days to predict into the future  
- **SMA/EMA Sliders:** Adjust window to see how moving averages react to different periods
""")

# ==========================
# Step 8: FAQ Section
# ==========================
st.header("❓ Frequently Asked Questions (FAQ)")

faq_list = [
    {
        "question": "1️⃣ Is this app accurate for predicting stock prices?",
        "answer": "No model can predict stock prices perfectly. This app uses a simple Linear Regression model for educational purposes. Predictions give an estimate based on historical patterns but cannot guarantee future stock prices."
    },
    {
        "question": "2️⃣ What model is used in this app?",
        "answer": "We use **Linear Regression**, a simple model that finds a relationship between historical stock features (Open, High, Low, Volume) and Close price."
    },
    {
        "question": "3️⃣ What do MAE, RMSE, and R² mean?",
        "answer": "- **MAE (Mean Absolute Error):** Average prediction error.\n- **RMSE (Root Mean Squared Error):** Highlights larger errors.\n- **R² Score:** How much variance in the historical Close price the model explains."
    },
    {
        "question": "4️⃣ Can I predict any stock?",
        "answer": "Yes, you can enter any valid stock ticker supported by Yahoo Finance, including international and Indian stocks (use .NS suffix for NSE stocks)."
    },
    {
        "question": "5️⃣ Can this app predict real-time stock movements?",
        "answer": "No, predictions are based on historical data up to the selected end date. Stock prices are highly volatile and influenced by real-time events, so future prices may vary."
    },
    {
        "question": "6️⃣ Why do we use Open, High, Low, and Volume as features?",
        "answer": "These are common indicators of stock behavior. The model uses them to learn patterns that help estimate the Close price."
    },
    {
        "question": "7️⃣ What other models could we use in the future?",
        "answer": "Advanced models include **LSTM**, **Random Forest**, **Gradient Boosting**, and other deep learning models that can capture more complex patterns in stock data."
    },
    {
        "question": "8️⃣ What is the difference between SMA and EMA?",
        "answer": "- **SMA (Simple Moving Average):** Average over a period, smooths fluctuations.\n- **EMA (Exponential Moving Average):** Gives more weight to recent prices and reacts faster to changes."
    },
    {
        "question": "9️⃣ What does a negative daily return mean?",
        "answer": "A negative daily return (e.g., -5%) means the stock's closing price **dropped** compared to the previous day. Positive values mean the price increased."
    },
    {
        "question": "🔟 Can I download the data and predictions?",
        "answer": "Yes! You can download both historical data and predicted Close prices as CSV files for further analysis."
    }
]

# -------------------------
# Display FAQ using expanders
# -------------------------
for faq in faq_list:
    with st.expander(faq["question"]):
        st.write(faq["answer"])
