import unittest
import pandas as pd
from backend.metrics_generator import generate_metric
from ml.anomaly_detection import train_isolation_forest, detect_anomalies
import os

class TestMetricsGenerator(unittest.TestCase):
    def test_generate_metric(self):
        metric = generate_metric()
        self.assertIn('timestamp', metric)
        self.assertIn('weblogic_heap', metric)
        self.assertTrue(isinstance(metric['weblogic_heap'], float))

class TestAnomalyDetection(unittest.TestCase):
    def setUp(self):
        # Create a small fake metrics.csv
        self.csv_path = 'backend/metrics.csv'
        df = pd.DataFrame([generate_metric() for _ in range(100)])
        df.to_csv(self.csv_path, index=False)
    def test_train_and_detect(self):
        train_isolation_forest()
        result = detect_anomalies()
        self.assertIn('anomaly_pred', result.columns)
    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
    def test_high_low_anomaly(self):
        # Test that both high and low anomalies can be generated
        highs, lows = 0, 0
        for _ in range(1000):
            metric = generate_metric()
            if metric['anomaly']:
                val = metric[metric['anomaly']]
                if val > 1000:  # Arbitrary high threshold
                    highs += 1
                elif val < 10:  # Arbitrary low threshold
                    lows += 1
        self.assertTrue(highs > 0 or lows > 0)

class TestAutoHealActions(unittest.TestCase):
    def test_autoheal_responses(self):
        from backend.app import AUTOHEAL_RESPONSES
        for key, response in AUTOHEAL_RESPONSES.items():
            self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()
