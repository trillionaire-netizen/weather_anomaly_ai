import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="AWS Intelligent Anomaly Dashboard", layout="wide")

st.title("🛰️ AI/ML Intelligent Anomaly Detection for Automatic Weather Stations")
st.markdown("**SIH26073 Solution Matrix** | Ministry of Earth Sciences (MoES) Ecosystem")

if st.button("🔄 Poll AWS Data & Run AI Diagnostics", type="primary"):
    with st.spinner("Ingesting telemetry stream and running Isolation Forest models..."):
        try:
            response = requests.get("http://127.0.0.1:8000/api/analyze-station")
            data = response.json()
            
            df_recent = pd.DataFrame(data["recent_telemetry"])
            df_anomalies = pd.DataFrame(data["anomaly_logs"])
            
            # Metrics Overview
            col1, col2, col3 = st.columns(3)
            col1.metric("Target Station ID", data["station_id"])
            col2.metric("Telemetry Records Scanned", data["total_readings_processed"])
            col3.metric("Anomalies / Faults Flagged", data["total_anomalies_flagged"], delta_color="inverse")
            
            st.markdown("---")
            
            # Interactive Multi-Axis Subplots Chart
            st.subheader("📈 Multi-Parameter Atmospheric Telemetry Trends")
            
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.08,
                subplot_titles=("Temperature (°C)", "Surface Pressure (hPa)", "Precipitation (mm) & Wind Speed (km/h)")
            )
            
            # Row 1: Temperature
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["temperature"], name="Temp (°C)", line=dict(color="#FF4B4B")), row=1, col=1)
            
            # Row 2: Pressure
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["pressure"], name="Pressure (hPa)", line=dict(color="#3366CC")), row=2, col=1)
            
            # Row 3: Precipitation & Wind Speed
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["precipitation"], name="Precipitation (mm)", line=dict(color="#00CC96")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["wind_speed"], name="Wind Speed (km/h)", line=dict(color="#AB63FA")), row=3, col=1)
            
            fig.update_layout(height=750, template="plotly_dark", showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly Log Section
            st.subheader("🚨 Flagged Weather Anomalies & Sensor Outliers")
            if not df_anomalies.empty:
                st.dataframe(df_anomalies[["time", "temperature", "pressure", "wind_speed", "precipitation", "status"]], use_container_width=True)
            else:
                st.success("All active sensor streams are operating within safe baseline parameters.")
                
        except Exception as e:
            st.error(f"Connection failed: Ensure the FastAPI backend is running. Error: {e}")