import pandas as pd
import os
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_analysis(reference_path, current_path=None, report_dir="reports"):
    """Compares baseline reference data with current data to calculate data drift."""
    print("Starting Evidently AI Drift Analysis...")
    
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference training dataset not found at: {reference_path}")
        
    # Load reference dataset (our training features)
    ref_df = pd.read_csv(reference_path)
    
    # Select only the features the model actually uses (pollutants and temporal features)
    features_to_monitor = [
        'pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide',
        'hour_sin', 'hour_cos',
        'pm2_5_roll_mean_24h', 'pm2_5_roll_std_24h', 'pm2_5_lag_1h', 'pm2_5_lag_24h',
        'pm10_roll_mean_24h', 'pm10_roll_std_24h', 'pm10_lag_1h', 'pm10_lag_24h',
        'nitrogen_dioxide_roll_mean_24h', 'nitrogen_dioxide_roll_std_24h', 'nitrogen_dioxide_lag_1h', 'nitrogen_dioxide_lag_24h',
        'sulphur_dioxide_roll_mean_24h', 'sulphur_dioxide_roll_std_24h', 'sulphur_dioxide_lag_1h', 'sulphur_dioxide_lag_24h',
        'carbon_monoxide_roll_mean_24h', 'carbon_monoxide_roll_std_24h', 'carbon_monoxide_lag_1h', 'carbon_monoxide_lag_24h'
    ]
    
    # Filter columns to only what we care about
    ref_df = ref_df[features_to_monitor]
    
    # If no current dataset is passed, simulate by splitting the dataset:
    # 70% Reference (baseline) vs 30% Current (simulated new live data)
    if current_path is None or not os.path.exists(current_path):
        print("Simulating live data drift by splitting historical dataset...")
        split_idx = int(len(ref_df) * 0.7)
        curr_df = ref_df.iloc[split_idx:].copy()
        ref_df = ref_df.iloc[:split_idx].copy()
        
        # Let's inject a tiny bit of simulated drift into PM2.5 in the current dataset 
        # to guarantee the report has interesting findings (e.g. simulating a smoggy week)
        curr_df['pm2_5'] = curr_df['pm2_5'] * 1.25
        curr_df['pm2_5_roll_mean_24h'] = curr_df['pm2_5_roll_mean_24h'] * 1.2
    else:
        curr_df = pd.read_csv(current_path)[features_to_monitor]
        
    print(f"Reference dataset size: {ref_df.shape}")
    print(f"Current dataset size: {curr_df.shape}")
    
    # Initialize Evidently Data Drift Report
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    # Execute drift analysis
    drift_report.run(reference_data=ref_df, current_data=curr_df)
    
    # Create reports folder
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Save HTML interactive dashboard report
    html_path = os.path.join(report_dir, "drift_report.html")
    drift_report.save_html(html_path)
    print(f"Successfully generated visual HTML drift report: {html_path}")
    
    # 2. Save JSON metrics for automated trigger pipelines
    json_path = os.path.join(report_dir, "drift_report.json")
    drift_report.save_json(json_path)
    print(f"Successfully generated JSON drift report: {json_path}")
    
    # 3. Read JSON to parse and display a summary in the console
    with open(json_path, "r") as f:
        report_data = json.load(f)
        
    metrics = report_data["metrics"][0]["result"]
    number_of_features = metrics["number_of_columns"]
    drifted_features = metrics["number_of_drifted_columns"]
    share_of_drifted_features = metrics["share_of_drifted_columns"]
    dataset_drift = metrics["dataset_drift"]
    
    print("\n--- 📊 Evidently AI Drift Analysis Summary ---")
    print(f"Total features analyzed: {number_of_features}")
    print(f"Drifted features: {drifted_features} ({share_of_drifted_features:.2%})")
    print(f"Overall Dataset Drift Detected: {dataset_drift}")
    
    if dataset_drift:
        print("🚨 ALERT: Significant data drift detected! Retraining is highly recommended.")
    else:
        print("✅ Success: Data distribution is stable.")
    print("-----------------------------------------------\n")
    
if __name__ == "__main__":
    run_drift_analysis(
        reference_path="data/processed/featured_aqi_data.csv"
    )
