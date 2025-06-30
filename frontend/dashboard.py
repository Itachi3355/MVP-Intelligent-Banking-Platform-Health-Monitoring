import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px
import numpy as np
from io import StringIO
# Ensure ml module is importable for Streamlit LSTM usage
import sys
import os as _os
sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))

BACKEND_URL = 'http://localhost:5000/api/metrics'
AUTOHEAL_URL = 'http://localhost:5000/api/autoheal'

st.set_page_config(page_title="Banking Platform Health Dashboard", layout="wide")
st.title("💡 Intelligent Banking Platform Health Monitoring")
st.caption("AI-driven, real-time predictive maintenance for banking infrastructure.")

# Add custom CSS for a modern look
st.markdown(
    '''<style>
    .main {
        background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5em 1.5em;
        margin-top: 0.5em;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .stDataFrame {
        background: #f1f5f9;
        border-radius: 8px;
    }
    .stAlert {
        border-radius: 8px;
        font-weight: bold;
    }
    .stSidebar {
        background: #f1f5f9;
    }
    </style>''', unsafe_allow_html=True)

placeholder = st.empty()

def fetch_metrics():
    try:
        resp = requests.get(BACKEND_URL)
        if resp.status_code == 200:
            return pd.read_json(StringIO(resp.text))
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
    return pd.DataFrame()

def trigger_autoheal(action):
    try:
        resp = requests.post(AUTOHEAL_URL, json={'action': action})
        if resp.status_code == 200:
            st.success(f"Auto-heal triggered: {action}")
    except Exception as e:
        st.error(f"Auto-heal error: {e}")

# --- TOOLTIP/INFO FOR METRICS ---
metric_info = {
    'weblogic_heap': 'WebLogic JVM Heap Usage (%)',
    'weblogic_threads': 'WebLogic Thread Count',
    'oracle_query_time': 'Oracle DB Query Time (ms)',
    'system_cpu': 'System CPU Usage (%)',
    'system_memory': 'System Memory Usage (%)',
    'system_disk': 'System Disk Usage (%)',
    'oracle_session_count': 'Oracle DB Session Count'
}

# --- METRIC FILTERING ---
all_metrics = ['weblogic_heap', 'weblogic_threads', 'oracle_query_time', 'system_cpu', 'system_memory', 'system_disk', 'oracle_session_count']

# Add detailed anomaly explanations and suggestions
ANOMALY_EXPLANATIONS = {
    'weblogic_heap': {
        'desc': 'WebLogic JVM Heap Usage is abnormal. High values may indicate memory leaks or inefficient code. Low values may indicate underutilization.',
        'suggestion': 'Check for memory leaks, optimize application code, or adjust JVM heap settings.'
    },
    'weblogic_threads': {
        'desc': 'WebLogic Thread Count is abnormal. High values may indicate thread leaks or high load. Low values may indicate idle system.',
        'suggestion': 'Investigate thread pool usage, check for stuck threads, or tune thread pool size.'
    },
    'oracle_query_time': {
        'desc': 'Oracle DB Query Time is abnormal. High values may indicate slow queries or DB contention. Low values may indicate low activity.',
        'suggestion': 'Analyze slow queries, check DB indexes, or monitor DB load.'
    },
    'system_cpu': {
        'desc': 'System CPU Usage is abnormal. High values may indicate CPU bottleneck. Low values may indicate idle system.',
        'suggestion': 'Check running processes, optimize workloads, or scale resources.'
    },
    'system_memory': {
        'desc': 'System Memory Usage is abnormal. High values may indicate memory leaks or insufficient memory. Low values may indicate underutilization.',
        'suggestion': 'Check memory allocation, optimize applications, or add more memory.'
    },
    'system_disk': {
        'desc': 'System Disk Usage is abnormal. High values may indicate disk space issues. Low values may indicate underutilization.',
        'suggestion': 'Check disk space, clean up unnecessary files, or expand disk capacity.'
    },
    'oracle_session_count': {
        'desc': 'Oracle DB Session Count is abnormal. High values may indicate too many concurrent sessions. Low values may indicate connection issues.',
        'suggestion': 'Monitor database connections, optimize session handling, or increase session limits.'
    }
}

def show_dashboard():
    df = fetch_metrics()
    if not df.empty:
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            options=all_metrics,
            default=all_metrics,
            help="Choose which metrics to visualize on the dashboard."
        )
        cols = st.columns(len(selected_metrics))
        # Ensure autohealed_timestamps is always defined
        if 'autoheal_log' not in st.session_state:
            st.session_state['autoheal_log'] = []
        autohealed_timestamps = [t for t, _ in st.session_state['autoheal_log']]
        for i, m in enumerate(selected_metrics):
            with cols[i]:
                st.markdown(f"**{metric_info[m]}**")
                st.caption(f"{m.replace('_', ' ').title()}" )
                base_fig = px.line(df, x='timestamp', y=m, title=None)
                # Add LSTM predictions if available and enough data
                try:
                    from ml.anomaly_detection import predict_lstm_forecast
                    if len(df[m]) > 20:
                        preds = predict_lstm_forecast(metric=m, lookback=10, steps=10)
                        # Use future timestamps for LSTM predictions
                        last_time = pd.to_datetime(df['timestamp'].iloc[-1])
                        freq = pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[-2]) if len(df['timestamp']) > 1 else pd.Timedelta(seconds=5)
                        pred_x = [last_time + freq * (i+1) for i in range(10)]
                        base_fig.add_scatter(x=pred_x, y=preds, mode='lines+markers', marker=dict(color='green'), name='LSTM Forecast')
                except Exception as e:
                    st.caption(f"LSTM prediction error: {e}")
                # Add auto-heal markers if any
                if autohealed_timestamps:
                    for ts in autohealed_timestamps:
                        if ts in list(df['timestamp']):
                            idx = list(df['timestamp']).index(ts)
                            val = df.iloc[idx][m]
                            base_fig.add_scatter(x=[ts], y=[val], mode='markers', marker=dict(color='red', size=12), name='Auto-Heal')
                st.plotly_chart(base_fig, use_container_width=True, key=f"plotly_{m}")
        # Show anomalies (detect both high and low)
        anomalies = df[df['anomaly'].notnull()]
        if not anomalies.empty:
            last_anomaly = anomalies.iloc[-1]
            unique_key = f"autoheal_{last_anomaly['anomaly']}_{last_anomaly['timestamp']}"
            metric = last_anomaly['anomaly']
            value = last_anomaly[metric]
            explanation = ANOMALY_EXPLANATIONS.get(metric, {'desc': 'Abnormal value detected.', 'suggestion': 'Investigate the root cause.'})
            error_message = f"Anomaly detected in **{metric}**: Value = {value:.2f}.\n\n{explanation['desc']}"
            st.error(error_message)
            st.success(f"**Suggestion:** {explanation['suggestion']}")
            if st.button("Trigger Auto-Heal", key=unique_key):
                trigger_autoheal(metric)
                st.session_state['autoheal_log'].append((last_anomaly['timestamp'], metric))
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button("Download Latest Metrics (CSV)", df.to_csv(index=False), file_name="metrics.csv")
    else:
        st.info("No metrics available yet.")

refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 2, 30, 5)
auto_refresh = st.sidebar.checkbox("Auto-refresh graphs", value=True)
if st.sidebar.button("Refresh now"):
    st.session_state['refresh_graphs'] = True

# Only auto-refresh the graphs, not the whole page
if auto_refresh or st.session_state.get('refresh_graphs', False):
    show_dashboard()
    st.session_state['refresh_graphs'] = False
    st.experimental_rerun() if auto_refresh and hasattr(st, 'experimental_rerun') else None
else:
    show_dashboard()

st.sidebar.title("Dashboard Help & Info")
st.sidebar.markdown("""
**How to use this dashboard:**
- The main area shows live system metrics as line charts (Heap, Threads, Query Time, CPU).
- Hover over metric names for more info.
- If an anomaly is detected (high or low), a warning appears below the charts.
- Click the **Trigger Auto-Heal** button to simulate an automated fix for the detected anomaly.
- The table below the charts shows the latest 20 metric records, including which metric (if any) was anomalous.
- Use the sidebar to adjust the refresh rate, manually refresh, or enable/disable auto-refresh.
- Download the latest metrics as CSV for offline analysis.

**Legend:**
- **Heap, Threads, Query Time, CPU**: Key health metrics from simulated banking systems.
- **Anomaly**: A metric value that is much higher or lower than normal, flagged by the generator.
- **Auto-Heal**: Simulated action to resolve the anomaly (for demo purposes).
""")
