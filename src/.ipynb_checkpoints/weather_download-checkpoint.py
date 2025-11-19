#!/usr/bin/env python3
"""
Weather Data Download Script
Downloads hourly weather data from Open-Meteo API for PJM load areas.
"""

# ======================================================================
# # Weather Data Download (UTC + ET)
#
# This notebook downloads hourly weather data from Open-Meteo API for multiple load areas across the PJM region.
# Data is retrieved in UTC and converted to Eastern Time (ET).
# ======================================================================

# ======================================================================
# ## 1. Import Required Libraries
# ======================================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import time
from dateutil.relativedelta import relativedelta
import warnings

# ======================================================================
# ## 2. Define Load Area Coordinates
#
# Define the latitude and longitude for each load area in the PJM region.
# ======================================================================

zone_coords = pd.DataFrame({
    'load_area': ['AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT', 'AP', 'BC', 'CE', 'DAY', 'DEOK',
                  'DOM', 'DPLCO', 'DUQ', 'EASTON', 'EKPC', 'JC', 'ME', 'OE', 'OVEC', 'PAPWR',
                  'PE', 'PEPCO', 'PLCO', 'PN', 'PS', 'RECO', 'SMECO', 'UGI', 'VMEU'],
    'lat': [39.45, 37.25, 38.45, 38.20, 39.90, 37.30, 40.80, 41.85, 39.75, 39.10,
            37.55, 38.90, 40.45, 39.55, 37.75, 40.35, 40.20, 41.10, 38.85, 40.70,
            40.00, 38.90, 40.95, 41.15, 40.75, 41.00, 38.40, 40.25, 37.30],
    'lon': [-74.50, -81.30, -81.60, -83.10, -82.90, -80.90, -79.95, -86.10, -84.20, -84.50,
            -77.45, -75.50, -79.90, -75.10, -84.30, -74.65, -76.00, -81.25, -82.85, -77.80,
            -75.20, -76.95, -77.40, -77.80, -74.15, -74.10, -76.70, -75.65, -76.00]
})

print(f"Total load areas: {len(zone_coords)}")
zone_coords.head()

# ======================================================================
# ## 3. Configure API Parameters
# ======================================================================

# Year range to download
years = list(range(2016, 2026))

# API endpoint
api = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to retrieve
hourly_vars = "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m"

# Create output directory
# Directory paths (relative to src/)
W_DIR = "../data/raw/weather"

# Create directories if they don't exist
os.makedirs(W_DIR, exist_ok=True)

print(f"Years to download: {years[0]} - {years[-1]}")
print(f"Variables: {hourly_vars}")

# ======================================================================
# ## 4. Define Helper Functions
#
# ### 4.1 Month Sequence Generator
# ======================================================================

def month_seq(year):
    """Generate first day of each month for a given year"""
    return [datetime(year, m, 1).date() for m in range(1, 13)]

# ======================================================================
# ### 4.2 Weather Data Fetcher
# ======================================================================

def fetch_month(lat, lon, start_date, end_date, tries=3):
    """Fetch weather data for a specific month and location"""
    url = (f"{api}?latitude={lat}&longitude={lon}&start_date={start_date}"
           f"&end_date={end_date}&hourly={hourly_vars}&timezone=UTC")

    for k in range(1, tries + 1):
        try:
            res = requests.get(url, timeout=60)
            if res.status_code == 200:
                j = res.json()
                if 'hourly' in j and 'time' in j['hourly']:
                    return pd.DataFrame({
                        'datetime_beginning_utc': j['hourly']['time'],
                        'temp': j['hourly']['temperature_2m'],
                        'humidity': j['hourly']['relative_humidity_2m'],
                        'dew_point': j['hourly']['dew_point_2m'],
                        'precip': j['hourly']['precipitation'],
                        'wind': j['hourly']['wind_speed_10m']
                    })
        except Exception as e:
            print(f"Error on attempt {k}: {e}")

        time.sleep(0.7 * k)

    return None

# ======================================================================
# ## 5. Download Weather Data
#
# This cell downloads weather data for all load areas and years.
# ======================================================================

for y in years:
    print(f"\n{'='*50}")
    print(f"Downloading year {y} ...")
    print(f"{'='*50}")
    out = []

    for idx, row in zone_coords.iterrows():
        la = row['load_area']
        lat = row['lat']
        lon = row['lon']

        print(f"Processing {la} ({idx+1}/{len(zone_coords)})...", end=' ')

        for m in month_seq(y):
            # Skip future months
            if m > datetime.now().date():
                continue

            # Calculate end of month
            nxt = m + relativedelta(months=1)
            m_end = min(nxt - timedelta(days=1), datetime.now().date())

            # Fetch weather data
            w = fetch_month(lat, lon, str(m), str(m_end))

            if w is not None:
                # ---- Convert UTC → ET ----
                # Add seconds if not present
                w['datetime_beginning_utc'] = w['datetime_beginning_utc'].apply(
                    lambda x: f"{x}:00" if len(x.split(':')) == 2 else x
                )

                # Convert to datetime in UTC then to ET
                utc_times = pd.to_datetime(w['datetime_beginning_utc'], format='%Y-%m-%dT%H:%M:%S')
                utc_times = utc_times.dt.tz_localize('UTC')
                et_times = utc_times.dt.tz_convert('America/New_York')

                w['datetime_beginning_ept'] = et_times.dt.strftime('%Y-%m-%d %H:%M:%S')
                w['load_area'] = la

                out.append(w)

            time.sleep(0.15)

        print("Done")

    # ---- Write CSV ----
    if out:
        df = pd.concat(out, ignore_index=True)
        df = df.sort_values(['datetime_beginning_utc', 'load_area'])

        output_file = os.path.join(W_DIR, f"weather_hourly_{y}.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {output_file} ({len(df):,} rows)")

print("\n" + "="*50)
print("Done! Files saved")
print("="*50)

# ======================================================================
# ## 6. Verify Downloaded Data
#
# Quick check of the downloaded files.
# ======================================================================

# List all downloaded files
weather_files = sorted([f for f in os.listdir(W_DIR) if f.endswith('.csv')])

print(f"Total files downloaded: {len(weather_files)}\n")

for file in weather_files:
    filepath = os.path.join(W_DIR, file)
    df = pd.read_csv(filepath)
    print(f"{file}: {len(df):,} rows, {df['load_area'].nunique()} load areas")

# Display sample of most recent file
if weather_files:
    print(f"\nSample from {weather_files[-1]}:")
    sample_df = pd.read_csv(os.path.join(W_DIR, weather_files[-1]))
    display(sample_df.head(10))
