import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import random
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import datetime

ALPHA_VANTAGE_API_KEY = 'apikey'

# --- Step 1: Technical Indicators ---
def add_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    return df

# --- Step 2: Fetch Fundamental Data ---
def fetch_fundamental_data(symbol):
    fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

    income_data, _ = fd.get_income_statement_quarterly(symbol)

    balance_data, _ = fd.get_balance_sheet_quarterly(symbol)

    income_data = income_data[['fiscalDateEnding', 'totalRevenue', 'netIncome']]
    balance_data = balance_data[['fiscalDateEnding', 'totalAssets', 'totalLiabilities']]
    fund = pd.merge(income_data, balance_data, on='fiscalDateEnding', how='inner')
    fund['fiscalDateEnding'] = pd.to_datetime(fund['fiscalDateEnding'])
    
    numeric_columns = ['totalRevenue', 'netIncome', 'totalAssets', 'totalLiabilities']
    for col in numeric_columns:
        fund[col] = pd.to_numeric(fund[col], errors='coerce')
    
    fund['asset_turnover'] = fund['totalRevenue'] / fund['totalAssets']
    fund['net_margin'] = fund['netIncome'] / fund['totalRevenue']
    
    fund = fund.dropna(subset=['asset_turnover', 'net_margin'])
    
    return fund

# --- Step 3: Merge Fundamental Data with Daily Data ---
def merge_fundamentals_to_daily(daily, fund):
    fund = fund.sort_values('fiscalDateEnding')
    daily = daily.copy()
    daily = daily.reset_index()
    daily = pd.merge_asof(
        daily.sort_values('date'),
        fund.sort_values('fiscalDateEnding'),
        left_on='date', right_on='fiscalDateEnding',
        direction='backward'
    )
    daily.drop(['fiscalDateEnding'], axis=1, inplace=True)

    daily = daily.set_index('date')
    return daily

# --- Step 4: Load and preprocess stock data ---
def load_stock_data(stock_symbol, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Attempting to download {stock_symbol} data from Alpha Vantage (attempt {attempt + 1}/{max_retries})...")
            if attempt > 0:
                delay = random.uniform(2, 5)
                print(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')
            data = data.sort_index()
            data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
            if data.empty:
                raise ValueError(f"No data downloaded for {stock_symbol}")

            data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
            }, inplace=True)
            data['Prev Close'] = data['Close'].shift(1)
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data['Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Return'].rolling(window=5).std()
            data['High_Low_Spread'] = data['High'] - data['Low']
            data['Volume_Change'] = data['Volume'].pct_change()
            data['Close_lag1'] = data['Close'].shift(1)
            data['Return_lag1'] = data['Return'].shift(1)
            data['DayOfWeek'] = data.index.dayofweek

            data = add_technical_indicators(data)
            data.dropna(inplace=True)
            if len(data) < 20:
                raise ValueError(f"Insufficient data for {stock_symbol}. Only {len(data)} days available.")
            print(f"Successfully downloaded {len(data)} days of data for {stock_symbol}")
            return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed for {stock_symbol}")
                raise
            continue

# --- Step 5: Modeling and Evaluation ---
def run_models(stock_symbol, start_date, end_date):

    data = load_stock_data(stock_symbol, start_date, end_date)

    print("Fetching fundamental data...")
    fund = fetch_fundamental_data(stock_symbol)
    data = merge_fundamentals_to_daily(data, fund)

    features = [
        'Prev Close', 'MA_5', 'MA_10', 'Return', 'Volatility', 'High_Low_Spread', 'Volume_Change',
        'Close_lag1', 'Return_lag1', 'DayOfWeek', 'RSI_14', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
        'totalRevenue', 'netIncome', 'totalAssets', 'totalLiabilities', 'asset_turnover', 'net_margin'
    ]

    features = [f for f in features if f in data.columns]
    X = data[features]
    y = data['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    # Metrics
    def print_metrics(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return mse, mae, r2
    print_metrics("Linear Regression", y_test, y_pred_lr)
    print_metrics("Random Forest", y_test, y_pred_rf)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Prices')
    plt.plot(y_test.index, y_pred_lr, label='Linear Regression', linestyle='--')
    plt.plot(y_test.index, y_pred_rf, label='Random Forest', linestyle=':')
    plt.title(f"{stock_symbol} - Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Predict next day
    latest_data = data.iloc[-1][features].values.reshape(1, -1)
    latest_data_scaled = scaler.transform(latest_data)
    next_day_pred_lr = linear_model.predict(latest_data_scaled)[0]
    next_day_pred_rf = rf_model.predict(latest_data_scaled)[0]
    print(f"Next Day Prediction (Linear Regression): {next_day_pred_lr:.2f} USD")
    print(f"Next Day Prediction (Random Forest): {next_day_pred_rf:.2f} USD")
    return next_day_pred_lr, next_day_pred_rf

# --- Main Execution ---
if __name__ == "__main__":
    stock_symbol = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    try:
        next_day_pred_lr, next_day_pred_rf = run_models(stock_symbol, start_date, end_date)
        print(f"Predicted next closing price for {stock_symbol} (Linear Regression): ${next_day_pred_lr:.2f}")
        print(f"Predicted next closing price for {stock_symbol} (Random Forest): ${next_day_pred_rf:.2f}")
    except Exception as e:
        print(f"Error running model: {e}")
