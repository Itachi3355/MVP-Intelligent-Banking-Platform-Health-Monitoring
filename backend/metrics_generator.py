import random
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime

METRIC_NAMES = [
    'weblogic_heap', 'weblogic_threads',
    'oracle_query_time', 'oracle_session_count',
    'system_cpu', 'system_memory', 'system_disk'
]

def generate_metric(prev_metric=None):
    # Simulate realistic metrics with random walk
    if prev_metric is None:
        prev_metric = {
            'weblogic_heap': 60,
            'weblogic_threads': 120,
            'oracle_query_time': 120,
            'oracle_session_count': 200,
            'system_cpu': 40,
            'system_memory': 70,
            'system_disk': 80
        }
    metric = {
        'timestamp': datetime.utcnow().isoformat(),
        'weblogic_heap': np.clip(prev_metric['weblogic_heap'] + random.gauss(0, 2), 30, 90),
        'weblogic_threads': int(np.clip(prev_metric['weblogic_threads'] + random.gauss(0, 5), 50, 250)),
        'oracle_query_time': np.clip(prev_metric['oracle_query_time'] + random.gauss(0, 5), 60, 200),
        'oracle_session_count': int(np.clip(prev_metric['oracle_session_count'] + random.gauss(0, 10), 80, 350)),
        'system_cpu': np.clip(prev_metric['system_cpu'] + random.gauss(0, 2), 10, 90),
        'system_memory': np.clip(prev_metric['system_memory'] + random.gauss(0, 2), 30, 95),
        'system_disk': np.clip(prev_metric['system_disk'] + random.gauss(0, 1), 60, 95)
    }
    # Inject high or low anomaly randomly
    if random.random() < 0.05:
        anomaly_metric = random.choice(METRIC_NAMES)
        if random.random() < 0.5:
            # High anomaly
            metric[anomaly_metric] = np.clip(metric[anomaly_metric] * random.uniform(1.5, 2.5), 0, 9999)
            metric['anomaly'] = anomaly_metric
        else:
            # Low anomaly
            metric[anomaly_metric] = np.clip(metric[anomaly_metric] * random.uniform(0.1, 0.5), 0, 9999)
            metric['anomaly'] = anomaly_metric
    else:
        metric['anomaly'] = None
    return metric

def metrics_generator(csv_path='backend/metrics.csv', interval=5):
    prev_metric = None
    while True:
        metric = generate_metric(prev_metric)
        prev_metric = metric
        df = pd.DataFrame([metric])
        df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)
        time.sleep(interval)

if __name__ == '__main__':
    print('Starting metrics generator...')
    metrics_generator()
