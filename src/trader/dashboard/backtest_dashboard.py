import pandas as pd
import streamlit as st

from trader.utils.metrics import calculate_performance

st.set_page_config(page_title="Backtest Results Dashboard", layout="wide")
st.title("📊 Backtest Performance Viewer")

uploaded_file = st.file_uploader("Upload equity curve CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    equity_curve = df.iloc[:, 0]  # assumes first column is portfolio value

    st.line_chart(equity_curve, use_container_width=True)

    st.subheader("Performance Metrics")
    metrics = calculate_performance(equity_curve)
    st.json(metrics)
else:
    st.info("Upload a CSV file with 'timestamp' and portfolio value column.")
