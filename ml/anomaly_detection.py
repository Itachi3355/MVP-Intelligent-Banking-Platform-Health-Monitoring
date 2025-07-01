import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense  # <-- Fixed import
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import joblib

METRICS_CSV = os.path.join(os.path.dirname(__file__), '../backend/metrics.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'isoforest_model.pkl')

FEATURES = [
    'weblogic_heap', 'weblogic_threads',
    'oracle_query_time', 'oracle_session_count',
    'system_cpu', 'system_memory', 'system_disk'
]

ARIMA_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'arima_{metric}.pkl')

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

def seasonal_decompose_metric(metric='system_cpu', model='additive', period=None):
    """
    Perform seasonal decomposition on a metric time series.
    Args:
        metric (str): The metric column to decompose.
        model (str): 'additive' or 'multiplicative'.
        period (int or None): The number of periods in a complete seasonal cycle. If None, will infer.
    Returns:
        dict: trend, seasonal, resid, observed (all as numpy arrays), and timestamps.
    """
    df = pd.read_csv(METRICS_CSV)
    if 'timestamp' not in df.columns:
        raise ValueError('timestamp column required for seasonal decomposition')
    ts = pd.Series(df[metric].values, index=pd.to_datetime(df['timestamp']))
    if period is None:
        # Try to infer period (e.g., daily, weekly, etc.)
        inferred = pd.infer_freq(ts.index)
        if inferred is not None:
            period = pd.Timedelta('1D') // pd.Timedelta(inferred)
        else:
            period = max(2, min(30, len(ts)//10))  # fallback
    result = seasonal_decompose(ts, model=model, period=int(period), extrapolate_trend='freq')
    return {
        'trend': result.trend.values,
        'seasonal': result.seasonal.values,
        'resid': result.resid.values,
        'observed': result.observed.values,
        'timestamp': ts.index.values
    }

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

def train_random_forest_failure_classifier(label_col='failure_label'):
    """Train a Random Forest classifier to predict failure types (demo)."""
    df = pd.read_csv(METRICS_CSV)
    # For demo, create a synthetic label if not present
    if label_col not in df.columns:
        np.random.seed(42)
        df[label_col] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])  # 5% failures
    X = df[FEATURES].fillna(0)
    y = df[label_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), 'rf_failure_classifier.pkl')
    joblib.dump(model, model_path)
    print('Random Forest failure classifier trained and saved.')

def predict_failure_classification():
    """Predict failure probability for the latest metrics using Random Forest."""
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), 'rf_failure_classifier.pkl')
    if not os.path.exists(model_path):
        train_random_forest_failure_classifier()
    model = joblib.load(model_path)
    df = pd.read_csv(METRICS_CSV)
    X = df[FEATURES].fillna(0)
    proba = model.predict_proba(X)[-1, 1]  # Probability of failure for latest row
    pred = model.predict(X)[-1]
    return {'failure_probability': proba, 'failure_predicted': bool(pred)}

def train_arima_forecast(metric='system_cpu', order=(2,1,2)):
    df = pd.read_csv(METRICS_CSV)
    data = df[metric].astype(float)
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    joblib.dump(model_fit, ARIMA_MODEL_PATH.format(metric=metric))
    print(f'ARIMA model for {metric} trained and saved.')

def predict_arima_forecast(metric='system_cpu', steps=10, order=(2,1,2)):
    model_path = ARIMA_MODEL_PATH.format(metric=metric)
    df = pd.read_csv(METRICS_CSV)
    data = df[metric].astype(float)
    if not os.path.exists(model_path):
        train_arima_forecast(metric, order)
    model_fit = joblib.load(model_path)
    forecast = model_fit.forecast(steps=steps)
    conf_int = model_fit.get_forecast(steps=steps).conf_int()
    return {
        'forecast': forecast.values,
        'lower': conf_int.iloc[:, 0].values,
        'upper': conf_int.iloc[:, 1].values
    }

if __name__ == '__main__':
    train_isolation_forest()
    print(detect_anomalies().tail())
