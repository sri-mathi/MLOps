import pandas as pd
import numpy as np
import joblib
import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest

def train_model(input_path, model_path, n_estimators=100, contamination=0.05):
    """Train Isolation Forest and log to MLflow."""
    # Force MLflow to record our name instead of the system username
    os.environ["LOGNAME"] = "srimathi"
    os.environ["USER"] = "srimathi"
    
    df = pd.read_csv(input_path)
    
    # Select only numeric features for training
    # Drop non-numeric or metadata columns
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'hour' if it exists (since we have sin/cos)
    if 'hour' in features:
        features.remove('hour')
        
    X = df[features]
    
    print(f"Training on features: {features}")
    
    # We set tracking URI to a local mlruns folder
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("AQI_Anomaly_Detection")
    
    with mlflow.start_run():
        # Initialize and train
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )
        
        model.fit(X)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        preds = model.predict(X)
        df['anomaly'] = preds
        df['anomaly_score'] = model.decision_function(X)
        
        # Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("contamination", contamination)
        
        # Log metrics
        anomaly_count = (preds == -1).sum()
        mlflow.log_metric("anomaly_count", anomaly_count)
        mlflow.log_metric("anomaly_rate", anomaly_count / len(df))
        
        # Save model locally and to MLflow
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Register the model in MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="AQI_Anomaly_Detector"
        )
        
        print(f"Model saved to {model_path} and logged to MLflow.")
        print(f"Detected {anomaly_count} anomalies out of {len(df)} samples.")

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest for AQI Anomaly Detection")
    parser.add_argument("--input", type=str, default="data/processed/featured_aqi_data.csv", help="Input features path")
    parser.add_argument("--output", type=str, default="models/model.joblib", help="Output model path")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected ratio of anomalies")
    args = parser.parse_args()

    train_model(
        input_path=args.input,
        model_path=args.output,
        n_estimators=args.n_estimators,
        contamination=args.contamination
    )

if __name__ == "__main__":
    main()
