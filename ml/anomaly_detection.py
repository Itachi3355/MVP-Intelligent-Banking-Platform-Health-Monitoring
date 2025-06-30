import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense  # <-- Fixed import
from prophet import Prophet

METRICS_CSV = os.path.join(os.path.dirname(__file__), '../backend/metrics.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'isoforest_model.pkl')

FEATURES = [
    'weblogic_heap', 'weblogic_threads',
    'oracle_query_time', 'oracle_session_count',
    'system_cpu', 'system_memory', 'system_disk'
]

def train_isolation_forest():
    df = pd.read_csv(METRICS_CSV)
    X = df[FEATURES].fillna(0)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    import joblib
    joblib.dump(model, MODEL_PATH)
    print('Isolation Forest model trained and saved.')

def train_lstm_forecast(metric='system_cpu', lookback=10, epochs=10):
    df = pd.read_csv(METRICS_CSV)
    data = df[metric].values.astype(np.float32)
    # Prepare sequences
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Build LSTM model
    model = keras.Sequential([
        LSTM(32, input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    # Save model
    model.save(os.path.join(os.path.dirname(__file__), f'lstm_{metric}.keras'))
    print(f'LSTM model for {metric} trained and saved.')

def predict_lstm_forecast(metric='system_cpu', lookback=10, steps=10):
    from tensorflow import keras
    import shutil
    df = pd.read_csv(METRICS_CSV)
    data = df[metric].values.astype(np.float32)
    model_path = os.path.join(os.path.dirname(__file__), f'lstm_{metric}.keras')
    for attempt in range(2):  # Try twice: original, then after retrain
        try:
            if not os.path.exists(model_path):
                train_lstm_forecast(metric, lookback)
            model = keras.models.load_model(model_path)
            # Use last lookback points to predict next steps
            input_seq = data[-lookback:]
            preds = []
            for _ in range(steps):
                x = input_seq[-lookback:].reshape((1, lookback, 1))
                pred = model.predict(x, verbose=0)[0, 0]
                preds.append(pred)
                input_seq = np.append(input_seq, pred)
            return preds
        except (OSError, tf.errors.DataLossError) as e:
            # Model file is likely corrupted, delete and retrain
            if os.path.exists(model_path):
                os.remove(model_path)
            train_lstm_forecast(metric, lookback)
    # If it still fails, return empty or raise
    return []

# Prophet model training
def train_prophet_forecast(metric='system_cpu'):
    """Train and save a Prophet model for the given metric."""
    df = pd.read_csv(METRICS_CSV)
    # Prophet expects columns 'ds' (datetime) and 'y' (value)
    if 'timestamp' not in df.columns:
        raise ValueError('timestamp column required for Prophet')
    prophet_df = pd.DataFrame({'ds': pd.to_datetime(df['timestamp']), 'y': df[metric].astype(float)})
    model = Prophet()
    model.fit(prophet_df)
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), f'prophet_{metric}.pkl')
    joblib.dump(model, model_path)
    print(f'Prophet model for {metric} trained and saved.')

def predict_prophet_forecast(metric='system_cpu', steps=10):
    """Return Prophet forecast and intervals for the next N steps."""
    import joblib
    df = pd.read_csv(METRICS_CSV)
    if 'timestamp' not in df.columns:
        raise ValueError('timestamp column required for Prophet')
    prophet_df = pd.DataFrame({'ds': pd.to_datetime(df['timestamp']), 'y': df[metric].astype(float)})
    model_path = os.path.join(os.path.dirname(__file__), f'prophet_{metric}.pkl')
    if not os.path.exists(model_path):
        train_prophet_forecast(metric)
    model = joblib.load(model_path)
    # Create future dataframe
    last_date = pd.to_datetime(df['timestamp']).max()
    freq = pd.infer_freq(pd.to_datetime(df['timestamp'])) or 'min'
    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)
    # Return only the forecasted steps (not the history)
    forecast = forecast.tail(steps)
    return {
        'yhat': forecast['yhat'].values,
        'yhat_lower': forecast['yhat_lower'].values,
        'yhat_upper': forecast['yhat_upper'].values,
        'ds': forecast['ds'].values
    }

def predict_lstm_forecast_with_interval(metric='system_cpu', lookback=10, steps=10, n_iter=30, dropout_rate=0.2):
    """LSTM prediction with MC dropout for interval estimation."""
    from tensorflow import keras
    df = pd.read_csv(METRICS_CSV)
    data = df[metric].values.astype(np.float32)
    model_path = os.path.join(os.path.dirname(__file__), f'lstm_{metric}.keras')
    if not os.path.exists(model_path):
        train_lstm_forecast(metric, lookback)
    # Load model and enable dropout at inference
    model = keras.models.load_model(model_path, compile=False)
    # Patch dropout to always be active
    for layer in model.layers:
        if hasattr(layer, 'rate'):
            layer.training = True
    input_seq = data[-lookback:]
    preds = np.zeros((n_iter, steps))
    for i in range(n_iter):
        seq = input_seq.copy()
        for j in range(steps):
            x = seq[-lookback:].reshape((1, lookback, 1))
            pred = model(x, training=True).numpy()[0, 0]
            preds[i, j] = pred
            seq = np.append(seq, pred)
    # Mean and 95% interval
    mean = preds.mean(axis=0)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return {'mean': mean, 'lower': lower, 'upper': upper}

# --- ENHANCED ANOMALY DETECTION: Flag both high and low outliers ---
def detect_anomalies():
    import joblib
    if not os.path.exists(MODEL_PATH):
        train_isolation_forest()
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(METRICS_CSV)
    X = df[FEATURES].fillna(0)
    df['anomaly_pred'] = model.predict(X)
    # -1 is anomaly, 1 is normal
    # Add column for anomaly type (high/low)
    for feature in FEATURES:
        mean = X[feature].mean()
        std = X[feature].std()
        df[f'{feature}_anomaly_type'] = None
        df.loc[(df['anomaly_pred'] == -1) & (df[feature] > mean + 2*std), f'{feature}_anomaly_type'] = 'high'
        df.loc[(df['anomaly_pred'] == -1) & (df[feature] < mean - 2*std), f'{feature}_anomaly_type'] = 'low'
    return df.tail(50)[['timestamp'] + FEATURES + ['anomaly_pred'] + [f'{f}_anomaly_type' for f in FEATURES]]

if __name__ == '__main__':
    train_isolation_forest()
    print(detect_anomalies().tail())
