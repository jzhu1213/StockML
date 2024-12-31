import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download stock data
stock = 'AAPL'
data = yf.download(stock, start='2024-01-01', end='2024-12-30')

# Feature: Previous day's closing price
data['Prev Close'] = data['Close'].shift(1)

# Remove rows with NaN values (first row will be NaN after shift)
data.dropna(inplace=True)

# Features and Target
X = data[['Prev Close']]  # Use previous day's price as feature
y = data['Close']  # Target: Today's closing price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--')
plt.title(f'{stock} Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
