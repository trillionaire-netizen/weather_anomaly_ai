import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="AWS Intelligent Anomaly Dashboard", layout="wide")

st.title("🛰️ AI/ML Intelligent Anomaly Detection for Automatic Weather Stations")
st.markdown("**SIH26073 Solution Matrix** | Ministry of Earth Sciences (MoES) Ecosystem")

# --- CITY TABS ---
tab_raipur, tab_europe = st.tabs(["📍 Raipur (Tikrapara)", "☀️ Southern Europe (Madrid)"])

def render_station_tab(city_key, tab_label):
    st.sidebar.header(f"Controls ({tab_label})")
    inject_anomaly = st.sidebar.checkbox(f"Simulate Severe Weather ({city_key})", value=False, key=f"sim_{city_key}")
    use_msl = st.sidebar.checkbox(f"Match Phone App - Sea Level Pressure ({city_key})", value=False, key=f"msl_{city_key}")
    
    # Optional toggle to turn auto-refresh on or off during your presentation
    auto_refresh = st.sidebar.toggle(f"Enable Live Auto-Refresh ({tab_label})", value=True, key=f"auto_{city_key}")

    # Define an auto-updating fragment block running every 10 seconds if enabled
    refresh_rate = timedelta(minutes=15) if auto_refresh else None

    @st.fragment(run_every=refresh_rate)
    def live_telemetry_block():
        try:
            url = f"http://127.0.0.1:8000/api/analyze-station?city={city_key}&simulate_fault={str(inject_anomaly).lower()}&use_msl={str(use_msl).lower()}"
            response = requests.get(url)
            data = response.json()
            
            if "SIMULATED" in data.get("mode", ""):
                st.warning(f"⚠️ Mode: {data['mode']} | Pressure Format: {data['pressure_mode']}")
            else:
                st.success(f"✅ Mode: {data['mode']} | Pressure Format: {data['pressure_mode']}")
            
            df_recent = pd.DataFrame(data["recent_telemetry"])
            df_anomalies = pd.DataFrame(data["anomaly_logs"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target Station ID", data["station_id"])
            col2.metric("Telemetry Records Scanned", data["total_readings_processed"])
            col3.metric("Anomalies / Faults Flagged", data["total_anomalies_flagged"], delta_color="inverse")
            
            st.markdown("---")
            
            st.subheader(f"📈 Multi-Parameter Atmospheric Telemetry Trends — {data['location']}")
            
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.08,
                subplot_titles=("Temperature (°C)", "Pressure (hPa)", "Precipitation (mm) & Wind Speed (km/h)")
            )
            
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["temperature"], name="Temp (°C)", line=dict(color="#FF4B4B")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["pressure"], name="Pressure (hPa)", line=dict(color="#3366CC")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["precipitation"], name="Precipitation (mm)", line=dict(color="#00CC96")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_recent["time"], y=df_recent["wind_speed"], name="Wind Speed (km/h)", line=dict(color="#AB63FA")), row=3, col=1)
            
            fig.update_layout(height=750, template="plotly_dark", showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')
            
            col_tbl_head, col_tbl_btn = st.columns([3, 1])
            with col_tbl_head:
                st.subheader("📋 Complete Telemetry Data Record Matrix")
            with col_tbl_btn:
                csv_data = df_recent.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"aws_{city_key}_telemetry.csv",
                    mime="text/csv",
                    width='stretch',
                    key=f"dl_{city_key}"
                )
                    
            st.dataframe(
                df_recent[["time", "temperature", "humidity", "pressure", "wind_speed", "precipitation", "status"]], 
                width='stretch',
                height=300
            )
            
            st.subheader("🚨 Flagged Weather Anomalies & Sensor Outliers")
            if not df_anomalies.empty:
                st.dataframe(df_anomalies[["time", "temperature", "pressure", "wind_speed", "precipitation", "status"]], width='stretch', key=f"anom_{city_key}")
            else:
                st.success("All active sensor streams are operating within safe baseline parameters.")
                
        except Exception as e:
            st.error(f"Connection failed: Ensure the FastAPI backend is running. Error: {e}")

    # Call the fragment function inside the tab
    live_telemetry_block()

with tab_raipur:
    render_station_tab("raipur", "Raipur (Tikrapara)")

with tab_europe:
    render_station_tab("europe", "Southern Europe (Madrid)")