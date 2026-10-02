import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Network Fault Detection",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Real-Time Network Fault Detection")
st.subheader("AI-Based Network Operations Monitoring Dashboard")

st.sidebar.header("Monitoring Settings")

samples = st.sidebar.slider(
    "Number of telemetry samples",
    100,
    1000,
    300
)

contamination = st.sidebar.slider(
    "Anomaly sensitivity",
    0.01,
    0.15,
    0.05
)

if st.button("Generate Telemetry & Detect Anomalies"):

    np.random.seed(42)

    df = pd.DataFrame({
        "CPU": np.random.normal(50, 12, samples),
        "Memory": np.random.normal(60, 10, samples),
        "Latency": np.random.normal(40, 8, samples),
        "Traffic": np.random.normal(500, 100, samples)
    })

    # Add simulated abnormal events
    fault_indices = np.random.choice(
        samples,
        size=max(1, int(samples * 0.03)),
        replace=False
    )

    df.loc[fault_indices, "Latency"] += 100
    df.loc[fault_indices, "CPU"] += 35

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    df["Prediction"] = model.fit_predict(
        df[["CPU", "Memory", "Latency", "Traffic"]]
    )

    df["Status"] = df["Prediction"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    anomalies = df[df["Status"] == "Anomaly"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", samples)

    col2.metric("Anomalies Detected", len(anomalies))

    col3.metric(
        "Anomaly Percentage",
        f"{len(anomalies) / samples * 100:.2f}%"
    )

    st.subheader("Network Telemetry")

    fig = px.line(
        df,
        y="Latency",
        title="Network Latency Monitoring"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Detection Results")

    st.dataframe(df, use_container_width=True)

    if len(anomalies) > 0:
        st.error("⚠️ Potential network anomalies detected!")
    else:
        st.success("Network appears normal.")

else:
    st.info(
        "Click the button in the sidebar area to generate "
        "simulated telemetry and run anomaly detection."
    )
