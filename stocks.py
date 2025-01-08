import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to create LSTM model
def create_lstm_model(units, activation, learning_rate):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation, input_shape=(1, 1)))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Function to predict future stock prices using LSTM model
def predict_future_lstm(model, data, num_predictions, scaling_factor):
    predictions = []

    # Get the last data point from the input data
    last_data_point = data[-1]

    for _ in range(num_predictions):
        # Predict the next time step
        prediction = model.predict(last_data_point.reshape(1, 1, 1))
        predictions.append(prediction[0, 0])

        # Update last_data_point to include the predicted value for the next iteration
        last_data_point = np.append(last_data_point[1:], prediction)

    # Inverse normalize the predictions
    predictions = np.array(predictions) * scaling_factor

    return predictions

# Function to load and preprocess stock data
def load_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Previous closing price
    data['Prev Close'] = data['Close'].shift(1)
    
    # Remove NaN values
    data.dropna(inplace=True)
    
    return data

# Main function to run both models
def run_models(stock_symbol, start_date, end_date):
    # Load data
    data = load_stock_data(stock_symbol, start_date, end_date)
    
    # Prepare data for Linear Regression model
    X = data[['Prev Close']]  # Use previous day's price as feature
    y = data['Close']  # Target: Today's closing price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)

    # Linear Regression MSE
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    print(f'Linear Regression Mean Squared Error: {mse_lr}')

    # Plot Linear Regression
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred_lr, label='Predicted (Linear Regression)', linestyle='--')
    plt.title(f'{stock_symbol} - Linear Regression Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Prepare data for LSTM model
    data_normalized = data['Close'].values.reshape(-1, 1) / np.max(data['Close'])
    train_size = int(len(data_normalized) * 0.8)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]

    # Hyperparameters for LSTM model
    lstm_units = 50
    lstm_activation = 'relu'
    learning_rate = 0.001
    epochs = 100
    batch_size = 32

    # Create and train LSTM model
    lstm_model = create_lstm_model(lstm_units, lstm_activation, learning_rate)
    lstm_model.fit(train_data[:-1].reshape(-1, 1, 1), train_data[1:], epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict using LSTM model
    lstm_predictions = lstm_model.predict(test_data[:-1].reshape(-1, 1, 1)).flatten()

    # Inverse normalize LSTM predictions
    lstm_predictions = lstm_predictions * np.max(data['Close'])

    # Plot LSTM predictions
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[train_size:], test_data[1:] * np.max(data['Close']), label='Actual')
    plt.plot(data.index[train_size:], lstm_predictions, label='Predicted (LSTM)', linestyle='--')
    plt.title(f'{stock_symbol} - LSTM Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    return mse_lr, lstm_predictions

# Run the models for a specific stock and date range
stock_symbol = 'AAPL'  # Choose your stock symbol here
start_date = '2024-12-01'
end_date = '2024-12-31'

mse_lr, lstm_predictions = run_models(stock_symbol, start_date, end_date)

print(f"Predicted stock prices (LSTM) for {stock_symbol}:")
for i, prediction in enumerate(lstm_predictions, start=1):
    print(f"Day {i}: {prediction:.2f}")
