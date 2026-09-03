from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import requests
from datetime import datetime

app = FastAPI(title="AWS Weather Anomaly Detection API", version="2.0")

# Station coordinates mapping including Southern Europe (Madrid)
LOCATIONS = {
    "raipur": {"lat": 21.2190, "lon": 81.6350, "name": "Tikrapara, Raipur, CG"},
    "europe": {"lat": 40.4169, "lon": -3.7034, "name": "Madrid, Spain (Mediterranean Heat Dome)"}
}

def fetch_aws_telemetry(city: str = "raipur", simulate_fault: bool = False, use_msl: bool = False):
    loc = LOCATIONS.get(city, LOCATIONS["raipur"])
    url = f"https://api.open-meteo.com/v1/forecast?latitude={loc['lat']}&longitude={loc['lon']}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,pressure_msl,wind_speed_10m,precipitation"
    
    response = requests.get(url).json()
    hourly = response.get("hourly", {})
    
    df = pd.DataFrame({
        "time": hourly.get("time", []),
        "temperature": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relative_humidity_2m", []),
        "surface_pressure": hourly.get("surface_pressure", []),
        "pressure_msl": hourly.get("pressure_msl", []),
        "wind_speed": hourly.get("wind_speed_10m", []),
        "precipitation": hourly.get("precipitation", [])
    })
    df = df.dropna()
    
    # Filter out future forecasts safely
    df["time"] = pd.to_datetime(df["time"])
    current_time = pd.Timestamp.now()
    filtered_df = df[df["time"] <= current_time]
    
    # --- SAFETY FALLBACK ---
    # If the time filter accidentally leaves 0 rows, use the full dataset
    if len(filtered_df) > 0:
        df = filtered_df
    
    # Choose active pressure metric
    df["pressure"] = df["pressure_msl" if use_msl else "surface_pressure"]
    
    # Inject anomaly if toggled
    if simulate_fault and not df.empty:
        df.iloc[-1, df.columns.get_loc("pressure")] = 800.0      
        df.iloc[-1, df.columns.get_loc("precipitation")] = 150.0  
        df.iloc[-1, df.columns.get_loc("wind_speed")] = 140.0    
        
    return df, loc["name"]

@app.get("/api/analyze-station")
def analyze_station(city: str = "raipur", simulate_fault: bool = False, use_msl: bool = False):
    df, location_name = fetch_aws_telemetry(city=city, simulate_fault=simulate_fault, use_msl=use_msl)
    
    df["temp_lag1"] = df["temperature"].shift(1)
    df["temp_rolling_mean"] = df["temperature"].rolling(window=3).mean()
    df["pressure_rolling_mean"] = df["pressure"].rolling(window=3).mean()
    
    df = df.dropna().reset_index(drop=True)
    
    features = df[[
        "temperature", 
        "temp_lag1", 
        "temp_rolling_mean", 
        "pressure", 
        "pressure_rolling_mean", 
        "humidity", 
        "wind_speed", 
        "precipitation"
    ]]
    
    model = IsolationForest(n_estimators=100, random_state=42)
    model.fit(features)
    
    df["anomaly_score"] = model.decision_function(features)
    
    THRESHOLD = -0.15
    df["status"] = df["anomaly_score"].apply(lambda x: "Anomaly Detected" if x < THRESHOLD else "Normal")
    
    anomalies_df = df[df["status"] == "Anomaly Detected"]
    
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M")
    if not anomalies_df.empty:
        anomalies_df["time"] = anomalies_df["time"].dt.strftime("%Y-%m-%dT%H:%M")

    return {
        "status": "success",
        "station_id": f"AWS-{city.upper()}-01",
        "location": location_name,
        "pressure_mode": "Mean Sea Level (MSLP)" if use_msl else "Raw Surface Pressure",
        "mode": "SIMULATED FAULT ACTIVE" if simulate_fault else "LIVE TELEMETRY NORMAL",
        "total_readings_processed": len(df),
        "total_anomalies_flagged": len(anomalies_df),
        "recent_telemetry": df.tail(100).to_dict(orient="records"),
        "anomaly_logs": anomalies_df.to_dict(orient="records")
    }