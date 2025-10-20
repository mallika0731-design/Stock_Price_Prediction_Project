import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def compute_moving_averages(df, windows=(50, 200)):
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
    return df

def compute_RSI(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    return df

def add_macd(df):
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def add_bollinger_bands(df, window=20):
    df['BB_MID'] = df['Close'].rolling(window=window).mean()
    df['BB_STD'] = df['Close'].rolling(window=window).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    return df

def arima_forecast(series, order=(1, 1, 1), steps=30):
    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=steps)
    return forecast

def prepare_ml_data(df, lookback=10, features=None):
    X, y = [], []
    if features is None:
        features = ['Close']
    data = df[features].values
    for i in range(lookback, len(data) - 1):
        X.append(data[i - lookback:i].flatten())
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def run_ml_forecast(df, model_cls, steps=30, lookback=10, features=None, **kwargs):
    X, y = prepare_ml_data(df, lookback, features=features)
    # Remove samples with NaN values to avoid errors
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = model_cls(**kwargs)
    model.fit(X_scaled, y)

    last_window_raw = df[features].values[-lookback:].flatten()
    # Make sure last_window has no NaN by filling or dropping; here fill with last valid
    mask_last = ~np.isnan(last_window_raw)
    if not np.all(mask_last):
        # Fill missing with last valid (simple forward fill)
        last_valid_index = np.where(mask_last)[0][-1]
        last_window_raw[np.isnan(last_window_raw)] = last_window_raw[last_valid_index]

    last_window = last_window_raw.reshape(1, -1)

    preds = []
    for _ in range(steps):
        last_scaled = scaler.transform(last_window)
        next_pred = model.predict(last_scaled)[0]
        preds.append(next_pred)
        last_window = np.roll(last_window, -len(features))
        last_window[0, -len(features):] = np.array([next_pred] * len(features))
    return preds, getattr(model, "feature_importances_", None)


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # Remove keyword argument and compute RMSE manually
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def find_valid_csvs(folder):
    required_cols = ['date', 'open', 'high', 'low', 'close']
    csv_files = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
    valid_files = []
    names = []
    for f in csv_files:
        try:
            tempdf = pd.read_csv(f, nrows=2)
            temp_cols = [c.lower().strip() for c in tempdf.columns]
            if all(col in temp_cols for col in required_cols):
                valid_files.append(f)
                names.append(os.path.splitext(os.path.basename(f))[0])
        except Exception as e:
            print(f"Warning: Skipping file {f} due to error: {e}")
    return names, valid_files

def main():
    print("===== NSE Stock Analytics Script =====")
    stocks_folder = r"C:\Users\Lenovo\Desktop\Datasets"
    print(f"Searching for CSV files in: {stocks_folder}")
    stock_names, valid_files = find_valid_csvs(stocks_folder)
    if not valid_files:
        print("No valid CSV stock price files found! Exiting.")
        return

    print(f"\nFound {len(stock_names)} stocks:")
    for idx, name in enumerate(stock_names):
        print(f"{idx}: {name}")

    while True:
        try:
            selected_idx = int(input(f"Enter stock index (0-{len(stock_names)-1}): ").strip())
            if 0 <= selected_idx < len(stock_names):
                break
            else:
                print("Index out of range. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    selected_stock = stock_names[selected_idx]
    file_path = valid_files[selected_idx]
    print(f"\nLoading data for stock: {selected_stock}")

    df = pd.read_csv(file_path)
    date_col = None
    for col in df.columns:
        if col.lower().strip() in ['date', 'timestamp']:
            date_col = col
            break
    if date_col is None:
        print(f"Error: No date-like column found in {selected_stock}. Columns: {df.columns.tolist()}")
        return

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    df = compute_moving_averages(df)
    df = compute_RSI(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)

    print("\nLast 5 rows:")
    print(df.tail())

    split_pct = 0.8
    split = int(len(df) * split_pct)
    df_train, df_test = df.iloc[:split], df.iloc[split:]
    ml_features = ['Close', 'MA50', 'MA200', 'MACD']

    print("\nRunning forecasts...")

    arima_pred = arima_forecast(df_train['Close'], steps=len(df_test))
    mae_arima, rmse_arima = calc_metrics(df_test['Close'].iloc[:len(arima_pred)], arima_pred)

    rf_pred, rf_imp = run_ml_forecast(
        df_train, RandomForestRegressor, n_estimators=100, steps=len(df_test),
        lookback=10, features=ml_features, random_state=42
    )
    mae_rf, rmse_rf = calc_metrics(df_test['Close'].iloc[:len(rf_pred)], rf_pred)

    svr_pred, _ = run_ml_forecast(
        df_train, SVR, steps=len(df_test), C=5, lookback=10, features=ml_features
    )
    mae_svr, rmse_svr = calc_metrics(df_test['Close'].iloc[:len(svr_pred)], svr_pred)

    min_len = min(len(arima_pred), len(rf_pred), len(svr_pred))
    ensemble_pred = (np.array(arima_pred[:min_len]) + np.array(rf_pred[:min_len]) + np.array(svr_pred[:min_len])) / 3
    mae_ens, rmse_ens = calc_metrics(df_test['Close'].iloc[:min_len], ensemble_pred)

    print("\nModel Performance Metrics:")
    print(f"ARIMA      - MAE: {mae_arima:.4f}, RMSE: {rmse_arima:.4f}")
    print(f"RandomForest - MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")
    print(f"SVR        - MAE: {mae_svr:.4f}, RMSE: {rmse_svr:.4f}")
    print(f"Ensemble   - MAE: {mae_ens:.4f}, RMSE: {rmse_ens:.4f}")

    # Combined plot
    plt.figure(figsize=(16,8))
    plt.plot(df.index, df['Close'], label='Close (Full)')
    if 'MA50' in df.columns:
        plt.plot(df.index, df['MA50'], '--', label='MA50')
    if 'MA200' in df.columns:
        plt.plot(df.index, df['MA200'], '--', label='MA200')

    # Forecast curves on test set only
    plt.plot(df_test.index[:min_len], df_test['Close'].iloc[:min_len], label='True Close (Test)', linewidth=3)
    plt.plot(df_test.index[:min_len], arima_pred[:min_len], label='ARIMA Forecast', linewidth=2)
    plt.plot(df_test.index[:min_len], rf_pred[:min_len], label='RF Forecast', linewidth=2)
    plt.plot(df_test.index[:min_len], svr_pred[:min_len], label='SVR Forecast', linewidth=2)
    plt.plot(df_test.index[:min_len], ensemble_pred, label='Ensemble', linewidth=2)

    plt.title(f"{selected_stock} - Prices, Moving Averages, Forecasts")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
