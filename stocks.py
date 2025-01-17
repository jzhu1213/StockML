import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Function to load and preproccess stock data
def load_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Add previous closing price
    data['Prev Close'] = data['Close'].shift(1)
    
    # Add moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    
    # Remove NaN values caused by rolling and shifting
    data.dropna(inplace=True)
    return data

# Train and evaluate the Linear Regression model
def run_model(stock_symbol, start_date, end_date):
    # Load data
    data = load_stock_data(stock_symbol, start_date, end_date)
    
    # Prepare features and target
    X = data[['Prev Close', 'MA_5', 'MA_10']]  # Features
    y = data['Close']  # Target
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    # Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression Mean Squared Error: {mse}")
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Prices')
    plt.plot(y_test.index, y_pred, label='Predicted Prices', linestyle='--')
    plt.title(f"{stock_symbol} - Linear Regression Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    return mse

# Run the model for a specific stock and date range
if __name__ == "__main__":
    stock_symbol = 'AAPL'  # Specify your stock symbol
    start_date = '2024-01-01' # Specify your start date
    end_date = '2024-12-30' # Specify your end date
    
    mse = run_model(stock_symbol, start_date, end_date)
