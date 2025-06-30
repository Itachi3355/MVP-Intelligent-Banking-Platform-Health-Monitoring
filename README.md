# Intelligent Banking Platform Health Monitoring & Predictive Maintenance System

This MVP project includes:
- Streamlit dashboard (frontend)
- Flask backend API (metrics, auto-heal simulation)
- Python metrics generator with anomaly injection
- ML anomaly detection (Isolation Forest)
- Unit tests

## Quick Start

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start the backend API:
   ```sh
   python backend/app.py
   ```
3. Start the Streamlit dashboard:
   ```sh
   streamlit run frontend/dashboard.py
   ```

## Project Structure
- `backend/` - Flask API for metrics and auto-heal
- `frontend/` - Streamlit dashboard
- `ml/` - ML models and anomaly detection
- `tests/` - Unit tests
- `.github/` - Copilot instructions

## Usage Instructions

### Local Development
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start the metrics generator:
   ```sh
   python backend/metrics_generator.py
   ```
3. In a new terminal, start the backend API:
   ```sh
   python backend/app.py
   ```
4. In another terminal, start the Streamlit dashboard:
   ```sh
   streamlit run frontend/dashboard.py
   ```

### Deployment
- **Streamlit Cloud**: Push your repo to GitHub, connect to Streamlit Cloud, and set the entrypoint to `frontend/dashboard.py`.
- **Render**: Use a Python web service for the backend and a separate Streamlit service for the dashboard.

### Authentication
- For demos, add a password prompt to the dashboard:
   ```python
   import streamlit as st
   if st.text_input('Password', type='password') != 'yourpassword':
       st.stop()
   ```

## Features
- Realistic metric simulation with both high and low anomalies
- Advanced ML anomaly detection (Isolation Forest, LSTM/Prophet ready)
- Interactive, filterable dashboard with tooltips and export
- Simulated auto-heal actions
- Unit tests for all modules

## Author
Murali Sai Srinivas Madhyahnapu

# MVP-Intelligent-Banking-Platform-Health-Monitoring