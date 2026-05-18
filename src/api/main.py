from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from prometheus_client import Counter, Gauge, make_asgi_app, Histogram
import time

app = FastAPI(
    title="AQI Real-Time Anomaly Detection Service",
    description="Production-grade API serving Isolation Forest for multivariate pollutant time series",
    version="1.0.0"
)

# Prometheus metrics
PREDICTION_COUNTER = Counter("aqi_predictions_total", "Total predictions served", ["city"])
ANOMALY_COUNTER = Counter("aqi_anomalies_total", "Total anomalies detected", ["city"])
LATENCY_HISTOGRAM = Histogram("aqi_prediction_latency_seconds", "Prediction latency in seconds")
MODEL_VERSION_GAUGE = Gauge("aqi_model_version", "Currently loaded model version")

# In-memory history cache to compute rolling windows and lags in real-time
# Key: city name, Value: List of dicts representing the raw pollutant readings
HISTORY_CACHE = {
    "Chennai": [],
    "Bangalore": [],
    "Delhi": []
}

CACHE_LIMIT = 24  # We need at least 24 hours of data for rolling 24h metrics and lag 24h

# Load model at startup
MODEL_PATH = "models/model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            # Set dummy version gauge to 3 since we just registered v3
            MODEL_VERSION_GAUGE.set(3)
            print("Successfully loaded Isolation Forest model from disk.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model path {MODEL_PATH} not found. Running in mock/cold-start mode.")

# Pydantic schema for input data validation
class PollutantReading(BaseModel):
    city: str = Field(..., description="One of Chennai, Bangalore, Delhi")
    timestamp: str = Field(..., description="ISO 8601 string, e.g. '2026-05-18T12:00:00'")
    pm2_5: float = Field(..., gt=0)
    pm10: float = Field(..., gt=0)
    nitrogen_dioxide: float = Field(..., gt=0)
    sulphur_dioxide: float = Field(..., gt=0)
    carbon_monoxide: float = Field(..., gt=0)

def compute_realtime_features(new_reading: PollutantReading):
    """Integrates the new reading, updates history cache, and computes features."""
    city = new_reading.city
    
    # Parse timestamp
    dt = datetime.fromisoformat(new_reading.timestamp)
    
    # 1. Calculate Cyclical Hour features
    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # 2. Update Cache
    cache = HISTORY_CACHE[city]
    new_data_point = new_reading.dict()
    new_data_point["hour_sin"] = hour_sin
    new_data_point["hour_cos"] = hour_cos
    cache.append(new_data_point)
    
    # Keep only the last 24 records
    if len(cache) > CACHE_LIMIT:
        cache.pop(0)
        
    df = pd.DataFrame(cache)
    
    # 3. Calculate Rolling Statistics & Lags
    features = {}
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]
    
    # Add base pollutants
    for p in pollutants:
        features[p] = new_reading.dict()[p]
        
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos
    
    # Calculate rolling and lag features
    for p in pollutants:
        features[f"{p}_roll_mean_24h"] = df[p].mean()
        features[f"{p}_roll_std_24h"] = df[p].std() if len(df) > 1 else 0.0
        features[f"{p}_lag_1h"] = df[p].iloc[-2] if len(df) > 1 else 0.0
        features[f"{p}_lag_24h"] = df[p].iloc[0] if len(df) == CACHE_LIMIT else 0.0
        
    # One-hot encoded cities columns
    for c in ["Chennai", "Bangalore", "Delhi"]:
        features[f"city_{c}"] = 1.0 if city == c else 0.0
        
    # Order features precisely as the model expects
    ordered_features = [
        'pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide',
        'hour_sin', 'hour_cos',
        'pm2_5_roll_mean_24h', 'pm2_5_roll_std_24h', 'pm2_5_lag_1h', 'pm2_5_lag_24h',
        'pm10_roll_mean_24h', 'pm10_roll_std_24h', 'pm10_lag_1h', 'pm10_lag_24h',
        'nitrogen_dioxide_roll_mean_24h', 'nitrogen_dioxide_roll_std_24h', 'nitrogen_dioxide_lag_1h', 'nitrogen_dioxide_lag_24h',
        'sulphur_dioxide_roll_mean_24h', 'sulphur_dioxide_roll_std_24h', 'sulphur_dioxide_lag_1h', 'sulphur_dioxide_lag_24h',
        'carbon_monoxide_roll_mean_24h', 'carbon_monoxide_roll_std_24h', 'carbon_monoxide_lag_1h', 'carbon_monoxide_lag_24h',
        'city_Bangalore', 'city_Chennai', 'city_Delhi'
    ]
    
    feature_vector = [features.get(f, 0.0) for f in ordered_features]
    return np.array([feature_vector])

@app.post("/predict")
def predict_anomaly(reading: PollutantReading):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    if reading.city not in HISTORY_CACHE:
        raise HTTPException(status_code=400, detail="Invalid city. Must be Chennai, Bangalore, or Delhi.")
        
    start_time = time.time()
    
    # Increment Prometheus request counter
    PREDICTION_COUNTER.labels(city=reading.city).inc()
    
    # 1. Feature Engineering
    feature_array = compute_realtime_features(reading)
    
    # 2. ML Prediction
    prediction = model.predict(feature_array)[0]
    score = model.decision_function(feature_array)[0]
    
    is_anomaly = True if prediction == -1 else False
    
    if is_anomaly:
        ANOMALY_COUNTER.labels(city=reading.city).inc()
        
    # Latency tracking
    latency = time.time() - start_time
    LATENCY_HISTOGRAM.observe(latency)
    
    return {
        "city": reading.city,
        "timestamp": reading.timestamp,
        "is_anomaly": is_anomaly,
        "anomaly_score": float(score),
        "status": "anomaly detected" if is_anomaly else "normal"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "active_model_version": 3
    }

# Expose Prometheus metrics endpoint to ASG
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
