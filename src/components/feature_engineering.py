import pandas as pd
import numpy as np
import os
import argparse

def create_features(df):
    """Perform feature engineering on AQI data."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['city', 'timestamp'])
    
    # Cyclical Encoding for Hour
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Rolling Statistics (24h window)
    for col in ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']:
        df[f'{col}_roll_mean_24h'] = df.groupby('city')[col].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
        df[f'{col}_roll_std_24h'] = df.groupby('city')[col].transform(lambda x: x.rolling(window=24, min_periods=1).std())
        
        # Lag features
        df[f'{col}_lag_1h'] = df.groupby('city')[col].shift(1)
        df[f'{col}_lag_24h'] = df.groupby('city')[col].shift(24)

    # Fill NaNs from rolling/lag operations
    df = df.fillna(0)
    
    # City One-Hot Encoding (optional for Isolation Forest, but helpful for context)
    df = pd.get_dummies(df, columns=['city'], prefix='city', dtype=int)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Process raw AQI data into features")
    parser.add_argument("--input", type=str, default="data/raw/aqi_data.csv", help="Input file path")
    parser.add_argument("--output", type=str, default="data/processed/featured_aqi_data.csv", help="Output file path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    featured_df = create_features(df)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    featured_df.to_csv(args.output, index=False)
    print(f"Processed features saved to {args.output}")
    print(f"Shape: {featured_df.shape}")

if __name__ == "__main__":
    main()
