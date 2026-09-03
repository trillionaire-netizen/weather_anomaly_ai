import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="AWS Intelligent Anomaly Dashboard", layout="wide")

st.title("🛰️ AI/ML Intelligent Anomaly Detection for Automatic Weather Stations")
st.markdown("**SIH26073 Solution Matrix** | Ministry of Earth Sciences (MoES) Ecosystem")

# --- UI CONTROLS IN SIDEBAR ---
st.sidebar.header("Demo Controls")
inject_anomaly = st.sidebar.checkbox("Simulate Severe Weather/Sensor Fault", value=False)
use_msl = st.sidebar.checkbox("Match Phone App (Use Sea Level Pressure)", value=False, 
                              help="Switches pressure from terrain surface (~971 hPa) to Mean Sea Level (~1007 hPa)")

if st.button("🔄 Poll AWS Data & Run AI Diagnostics", type="primary"):
    with st.spinner("Ingesting telemetry stream and running AI models..."):
        try:
            # Pass both toggle parameters to the backend
            url = f"http://127.0.0.1:8000/api/analyze-station?simulate_fault={str(inject_anomaly).lower()}&use_msl={str(use_msl).lower()}"
            response = requests.get(url)
            data = response.json()
            
            # Show status banners
            if "SIMULATED" in data.get("mode", ""):
                st.warning(f"⚠️ Mode: {data['mode']} | Pressure Format: {data['pressure_mode']}")
            else:
                st.success(f"✅ Mode: {data['mode']} | Pressure Format: {data['pressure_mode']}")
            
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
            
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["temperature"], name="Temp (°C)", line=dict(color="#FF4B4B")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["pressure"], name="Pressure (hPa)", line=dict(color="#3366CC")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["precipitation"], name="Precipitation (mm)", line=dict(color="#00CC96")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["wind_speed"], name="Wind Speed (km/h)", line=dict(color="#AB63FA")), row=3, col=1)
            
            fig.update_layout(height=750, template="plotly_dark", showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')
            
            # --- TABULAR FORM & CSV EXPORT ---
            col_tbl_head, col_tbl_btn = st.columns([3, 1])
            with col_tbl_head:
                st.subheader("📋 Complete Telemetry Data Record Matrix")
            with col_tbl_btn:
                csv_data = df_recent.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="aws_raipur_telemetry.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
            st.markdown("Raw processed records fetched from the Raipur station feed:")
            st.dataframe(
                df_recent[["time", "temperature", "humidity", "pressure", "wind_speed", "precipitation", "status"]], 
                width='stretch',
                height=300
            )
            
            # Anomaly Log Section
            st.subheader("🚨 Flagged Weather Anomalies & Sensor Outliers")
            if not df_anomalies.empty:
                st.dataframe(df_anomalies[["time", "temperature", "pressure", "wind_speed", "precipitation", "status"]], width='stretch')
            else:
                st.success("All active sensor streams are operating within safe baseline parameters.")
                
        except Exception as e:
            st.error(f"Connection failed: Ensure the FastAPI backend is running. Error: {e}")