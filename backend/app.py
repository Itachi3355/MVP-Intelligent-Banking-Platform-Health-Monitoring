import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, jsonify, request
import pandas as pd
import os
import datetime
from functools import wraps

app = Flask(__name__)

METRICS_CSV = os.path.join(os.path.dirname(__file__), 'metrics.csv')
DECOMP_LOG = os.path.join(os.path.dirname(__file__), 'decomp_requests.log')

AUTOHEAL_RESPONSES = {
    'weblogic_heap': 'Restart WebLogic JVM',
    'weblogic_threads': 'Scale up WebLogic thread pool',
    'oracle_query_time': 'Optimize Oracle DB query',
    'oracle_session_count': 'Clear Oracle DB sessions',
    'system_cpu': 'Restart system service',
    'system_memory': 'Clear system cache',
    'system_disk': 'Clean up disk space',
}

ROLES_PERMISSIONS = {
    'admin': {'decompose', 'autoheal', 'view'},
    'analyst': {'decompose', 'view'},
    'viewer': {'view'}
}

def send_alert(message, channel='slack'):
    """
    Demo alert integration. In production, replace with real API calls to Slack, JIRA, or ITSM.
    """
    alert_log = os.path.join(os.path.dirname(__file__), 'alerts.log')
    with open(alert_log, 'a') as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {channel.upper()} | {message}\n")
    print(f"[ALERT][{channel.upper()}] {message}")

def require_role(permission):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            role = request.headers.get('X-User-Role', 'viewer')
            if permission not in ROLES_PERMISSIONS.get(role, set()):
                with open('audit.log', 'a') as audit:
                    audit.write(f"{datetime.datetime.now().isoformat()} | {role} | DENIED | {permission} | {request.path}\n")
                return jsonify({'error': f'Permission denied for role: {role}'}), 403
            with open('audit.log', 'a') as audit:
                audit.write(f"{datetime.datetime.now().isoformat()} | {role} | ALLOWED | {permission} | {request.path}\n")
            return f(*args, **kwargs)
        return wrapper
    return decorator

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not os.path.exists(METRICS_CSV):
        return jsonify([])
    df = pd.read_csv(METRICS_CSV)
    # Return last 50 rows
    return df.tail(50).to_json(orient='records')

@app.route('/api/autoheal', methods=['POST'])
@require_role('autoheal')
def autoheal():
    data = request.json
    action = data.get('action', 'noop')
    response = AUTOHEAL_RESPONSES.get(action, 'No action mapped')
    print(f"[AUTOHEAL] Action triggered: {action} -> {response}")
    return jsonify({'status': 'success', 'action': action, 'response': response})

@app.route('/api/seasonal_decompose', methods=['GET'])
@require_role('decompose')
def api_seasonal_decompose():
    metric = request.args.get('metric', 'system_cpu')
    model = request.args.get('model', 'additive')
    period = request.args.get('period', None)
    # Log the request for audit/compliance
    with open(DECOMP_LOG, 'a') as logf:
        logf.write(f"{datetime.datetime.now().isoformat()} | metric={metric} | model={model} | period={period}\n")
    try:
        period = int(period) if period is not None else None
        # Check for enough data points
        df = pd.read_csv(METRICS_CSV)
        if len(df) < 10:
            return jsonify({'error': 'Not enough data points for decomposition (min 10 required).'}), 400
        from ml.anomaly_detection import seasonal_decompose_metric
        result = seasonal_decompose_metric(metric=metric, model=model, period=period)
        # --- Alert if strong seasonality detected ---
        import numpy as np
        seasonal = np.array(result['seasonal'])
        if len(seasonal) > 2:
            acf = np.correlate(seasonal - np.mean(seasonal), seasonal - np.mean(seasonal), mode='full')
            acf = acf[acf.size // 2:]
            peak_lag = np.argmax(acf[1:20]) + 1
            if peak_lag > 1 and acf[peak_lag] > 0.5 * acf[0]:
                send_alert(f"Strong seasonality detected for {metric}: cycle length ≈ {peak_lag} time steps.", channel='slack')
        for k in result:
            result[k] = result[k].tolist()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

from ml.anomaly_detection import predict_arima_forecast

@app.route('/api/arima_forecast', methods=['GET'])
@require_role('decompose')
def api_arima_forecast():
    metric = request.args.get('metric', 'system_cpu')
    steps = int(request.args.get('steps', 10))
    try:
        result = predict_arima_forecast(metric=metric, steps=steps)
        # Convert numpy arrays to lists for JSON serialization
        for k in result:
            result[k] = result[k].tolist()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
