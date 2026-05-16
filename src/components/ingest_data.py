import requests
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta

def fetch_aqi_data(city, lat, lon, days=7):
    """Fetch AQI data from Open-Meteo for a specific city."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    
    print(f"Fetching data for {city} ({lat}, {lon}) from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly_data = data.get("hourly", {})
        df = pd.DataFrame(hourly_data)
        df["city"] = city
        df["timestamp"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)
        return df
    else:
        print(f"Failed to fetch data for {city}: {response.status_code}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Ingest AQI data from Open-Meteo API")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data to fetch")
    parser.add_argument("--output", type=str, default="data/raw/aqi_data.csv", help="Output file path")
    args = parser.parse_args()

    cities = {
        "Chennai": (13.0827, 80.2707),
        "Bangalore": (12.9716, 77.5946),
        "Delhi": (28.6139, 77.2090)
    }

    all_data = []
    for city, coords in cities.items():
        df = fetch_aqi_data(city, coords[0], coords[1], days=args.days)
        if df is not None:
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        final_df.to_csv(args.output, index=False)
        print(f"Successfully saved data to {args.output}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
