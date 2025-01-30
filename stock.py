import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LSTM  # type: ignore
import matplotlib.pyplot as plt

# Set up Streamlit page configuration
st.set_page_config(page_title="Stock Prediction and Analysis", layout="wide")

# App Title
st.title("📈 Stock Prediction and Analysis")

# Sidebar inputs
st.sidebar.header("Input Parameters")
stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., KO, TSLA, GOOGL)", value="KO")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# Download Stock Data
@st.cache
def download_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

try:
    data = download_stock_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.error("No data found for the given symbol or date range. Please try again.")
    else:
        st.subheader(f"Stock Data for {stock_symbol}")
        st.write(data.head())

        # Plot Stock Closing Price
        st.subheader(f"Closing Price for {stock_symbol}")
        st.line_chart(data["Close"])

        # Preprocess data for LSTM
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        # Create dataset function
        def create_dataset(data, time_step=60):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM Model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(units=50, return_sequences=False),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        if st.sidebar.button("Train Model"):
            with st.spinner("Training the model... This may take some time."):
                model.fit(X_train, y_train, epochs=10, batch_size=22)
                st.success("Model training completed!")

            # Predict and inverse scale predictions
            predicted_stock_price = model.predict(X_test)
            predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Plot predictions vs actual
            st.subheader(f"Predicted vs Actual Stock Price for {stock_symbol}")
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
            plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
            plt.title(f"{stock_symbol} Stock Price Prediction")
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            st.pyplot(plt)

        # Additional Analysis (Histograms and Pie Chart)
        st.subheader("Stock Comparison")
        comparison_stocks = st.multiselect("Select Stocks to Compare", ["TSLA", "GOOGL", "NFLX", "MSFT", "MCD"], default=["TSLA", "GOOGL"])

        if comparison_stocks:
            comparison_data = {stock: yf.download(stock, start="2020-01-01", end="2025-01-01")['Close'] for stock in comparison_stocks}

            # Plot Histograms
            plt.figure(figsize=(10, 6))
            for stock, prices in comparison_data.items():
                plt.hist(prices, bins=50, alpha=0.5, label=stock)
            plt.legend()
            plt.title("Stock Price Distributions")
            st.pyplot(plt)

            # Calculate Percentage Changes
            percentage_changes = {}
            for stock, prices in comparison_data.items():
                # Extract the first and last closing prices as floats
                start_price = float(prices.iloc[0])  # First day's closing price
                end_price = float(prices.iloc[-1])   # Last day's closing price
                
                # Calculate the percentage change
                percentage_changes[stock] = ((end_price - start_price) / start_price) * 100

            # Display Percentage Changes
            st.subheader("Percentage Changes")
            st.write(percentage_changes)

            # Pie Chart
            labels = list(percentage_changes.keys())
            sizes = list(percentage_changes.values())
            colors = ['gold', 'lightblue', 'lightcoral', 'lightgreen', 'cyan']
            explode = [0.1 if i == max(sizes) else 0 for i in sizes]

            plt.figure(figsize=(8, 8))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.title("Stock Performance Comparison")
            st.pyplot(plt)

