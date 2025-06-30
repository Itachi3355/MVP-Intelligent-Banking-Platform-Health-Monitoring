from flask import Flask, jsonify, request
import pandas as pd
import os

app = Flask(__name__)

METRICS_CSV = os.path.join(os.path.dirname(__file__), 'metrics.csv')

AUTOHEAL_RESPONSES = {
    'weblogic_heap': 'Restart WebLogic JVM',
    'weblogic_threads': 'Scale up WebLogic thread pool',
    'oracle_query_time': 'Optimize Oracle DB query',
    'oracle_session_count': 'Clear Oracle DB sessions',
    'system_cpu': 'Restart system service',
    'system_memory': 'Clear system cache',
    'system_disk': 'Clean up disk space',
}

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not os.path.exists(METRICS_CSV):
        return jsonify([])
    df = pd.read_csv(METRICS_CSV)
    # Return last 50 rows
    return df.tail(50).to_json(orient='records')

@app.route('/api/autoheal', methods=['POST'])
def autoheal():
    data = request.json
    action = data.get('action', 'noop')
    response = AUTOHEAL_RESPONSES.get(action, 'No action mapped')
    print(f"[AUTOHEAL] Action triggered: {action} -> {response}")
    return jsonify({'status': 'success', 'action': action, 'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
