#!/usr/bin/env python3
"""
Weather Forecast Data Download and Prediction Generation

This script:
1. Downloads weather forecast data from Open-Meteo API
2. Creates features from weather data
3. Loads trained models
4. Generates predictions for tomorrow
5. Saves predictions in CSV format
"""

import sys
import os
import warnings

# Suppress all warnings and debug output
warnings.filterwarnings('ignore')
sys.stderr = open('/dev/null', 'w')

# ======================================================================
# # Weather Forecast Data Download (Nov 20-29, 2025)
#
# This notebook downloads hourly weather **forecast** data from Open-Meteo API for multiple load areas across the PJM region.
# Data is retrieved in UTC and converted to Eastern Time (ET).
#
# **Date Range**: November 20, 2025 00:00 UTC to November 29, 2025 23:00 UTC
# **Note**: Uses forecast API for future dates
# ======================================================================

# ======================================================================
# ## 1. Import Required Libraries
# ======================================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import holidays
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import warnings
import pickle

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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


# ======================================================================
# ## 3. Configure API Parameters
# ======================================================================

# Date range to download - automatically set to tomorrow
tomorrow = datetime.now() + timedelta(days=2)
start_date = "2025-11-20"
end_date = "2025-11-30"

# API endpoint - using forecast API for future dates
api = "https://api.open-meteo.com/v1/forecast"

# Weather variables to retrieve
hourly_vars = "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m"

# ======================================================================
# ## 4. Define Weather Data Fetcher Function
# ======================================================================

def fetch_weather_data(lat, lon, start_date, end_date, tries=3):
    """Fetch weather data for a specific date range and location"""
    # For forecast API, we use forecast_days parameter (max 16 days)
    url = (f"{api}?latitude={lat}&longitude={lon}"
           f"&hourly={hourly_vars}&timezone=UTC&forecast_days=16")

    for k in range(1, tries + 1):
        try:
            res = requests.get(url, timeout=60)
            if res.status_code == 200:
                j = res.json()
                if 'hourly' in j and 'time' in j['hourly']:
                    df = pd.DataFrame({
                        'datetime_beginning_utc': j['hourly']['time'],
                        'temp': j['hourly']['temperature_2m'],
                        'humidity': j['hourly']['relative_humidity_2m'],
                        'dew_point': j['hourly']['dew_point_2m'],
                        'precip': j['hourly']['precipitation'],
                        'wind': j['hourly']['wind_speed_10m']
                    })
        
                    # Filter to the specific date range we want
                    df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
                    mask = (df['datetime_beginning_utc'] >= start_date) & (df['datetime_beginning_utc'] <= end_date + ' 23:00:00')
                    df = df[mask].copy()
                    df['datetime_beginning_utc'] = df['datetime_beginning_utc'].astype(str)
        
                    return df
        except Exception as e:
            if k < tries:
                time.sleep(0.7 * k)

    return None

# ======================================================================
# ## 5. Download Weather Data for All Load Areas
#
# This cell downloads weather data for the specified date range across all load areas.
# ======================================================================

# Initialize list to collect data from all zones
all_data = []

for idx, row in zone_coords.iterrows():
    zone = row['load_area']
    lat = row['lat']
    lon = row['lon']

    # Fetch data for this zone
    df = fetch_weather_data(lat, lon, start_date, end_date)

    if df is not None and not df.empty:
        # Add load area identifier
        df['load_area'] = zone
        all_data.append(df)
    else:
        pass  # or handle the error
    # Small delay between requests to be respectful to the API
    time.sleep(0.5)


# ======================================================================
# ## 6. Combine Data and Convert to Eastern Time
#
# Combine all load area data and convert UTC timestamps to Eastern Time.
# ======================================================================

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

# Convert UTC to Eastern Time
combined_df['datetime_beginning_utc'] = pd.to_datetime(combined_df['datetime_beginning_utc'])
# Create Eastern Time column
utc = pytz.UTC
eastern = pytz.timezone('US/Eastern')
et_time = combined_df['datetime_beginning_utc'].dt.tz_localize(utc).dt.tz_convert(eastern)

# Format datetime in M/D/YYYY H:MM:SS AM/PM format
# Remove timezone info and format manually for cross-platform compatibility
et_time_no_tz = et_time.dt.tz_localize(None)

# Try Unix format first, fall back to Windows format if needed
try:
    combined_df['datetime_beginning_ept'] = et_time_no_tz.dt.strftime('%-m/%-d/%Y %-I:%M:%S %p')
except:
    # Windows format
    combined_df['datetime_beginning_ept'] = et_time_no_tz.dt.strftime('%#m/%#d/%Y %#I:%M:%S %p')

# Remove UTC column and reorder
combined_df = combined_df.drop('datetime_beginning_utc', axis=1)
column_order = ['datetime_beginning_ept', 'load_area', 'temp', 'humidity', 'dew_point', 'precip', 'wind']
combined_df = combined_df[column_order]

# ============================================================================
# CONFIGURATION
# ============================================================================
# Get absolute path to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is one level up from src/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Directory paths - absolute paths work everywhere
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target 29 load areas
KEEP_AREAS = [
    "AECO", "AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT", "AP", "BC", "CE", "DAY", "DEOK",
    "DOM", "DPLCO", "DUQ", "EASTON", "EKPC", "JC", "ME", "OE", "OVEC", "PAPWR",
    "PE", "PEPCO", "PLCO", "PN", "PS", "RECO", "SMECO", "UGI", "VMEU"
]

# Best Model Finding dates (dynamically set to tomorrow)
tomorrow_datetime = datetime.now() + timedelta(days=1)
TEST_START = tomorrow_datetime.strftime('%Y-%m-%d') + ' 00:00:00'
TEST_END = tomorrow_datetime.strftime('%Y-%m-%d') + ' 23:00:00'

test_start = pd.to_datetime(TEST_START)
test_end = pd.to_datetime(TEST_END)

# Rolling window for peak days
WINDOW_SIZE = 10  # days
NUM_PEAK_DAYS = 2

df= combined_df.copy()
def parse_et(series):
    """Parse datetime with Eastern Time timezone handling"""
    if pd.api.types.is_datetime64_any_dtype(series):
        result = pd.to_datetime(series)
        if result.dt.tz is None:
            return result.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
        return result
    result = pd.to_datetime(series, format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors='coerce')
    return result.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')

df = df.rename(columns={
    'datetime_beginning_ept': 'datetime',
    'load_area': 'region',
    'temp': 'temperature',
    'precip': 'precipitation',
    'wind': 'wind_speed'
})

# Keep only necessary columns
keep_cols = ['datetime', 'region', 'temperature', 'humidity', 'precipitation', 'wind_speed']
df = df[keep_cols]

df['datetime'] = parse_et(df['datetime'])
df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

def add_features(df):
    """Add temporal and calendar features"""
    df = df.copy()

    # ===== TEMPORAL FEATURES =====
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['datetime'].dt.month
    df['day_of_month'] = df['datetime'].dt.day
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ===== HOLIDAY FEATURES =====
    # Create US holiday calendar
    us_holidays = holidays.US(years=range(2016, 2026))
    df['date'] = df['datetime'].dt.date
    df['is_holiday'] = df['date'].apply(lambda x: int(x in us_holidays))

    # Day before/after holiday
    df['is_day_before_holiday'] = df['is_holiday'].shift(-24).fillna(0).astype(int)
    df['is_day_after_holiday'] = df['is_holiday'].shift(24).fillna(0).astype(int)

    # Thanksgiving - Cooking load, midday peak
    df['is_thanksgiving'] = df['date'].apply(
        lambda x: int(1 if us_holidays.get(x) == 'Thanksgiving' else 0)
    )

    # Christmas - Low commercial, high residential heating
    df['is_christmas'] = df['date'].apply(
        lambda x: int(1 if us_holidays.get(x) == 'Christmas Day' else 0)
    )

    # New Year's Day - Late night/early morning shift
    df['is_new_years'] = df['date'].apply(
        lambda x: int(1 if us_holidays.get(x) == "New Year's Day" else 0)
    )

    # July 4th - Summer, outdoor, evening grilling/fireworks
    df['is_july4'] = df['date'].apply(
        lambda x: int(1 if us_holidays.get(x) == 'Independence Day' else 0)
    )
    # Others
    df['is_other_holiday'] = (df['is_holiday']-df['is_thanksgiving']-df['is_christmas']-df['is_new_years']-df['is_july4'])

    return df

df = add_features(df)

# ======================================================================
# # Task 1
# ======================================================================

FEATURE_COLS = [
    # Temporal
    'hour', 'day_of_week', 'month', 'day_of_month', 'day_of_year', 'week_of_year', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    # Weather
    'temperature', 'humidity', 'wind_speed', 'precipitation',
    # 'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
    'is_thanksgiving', 'is_christmas', 'is_new_years', 'is_july4', 'is_other_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
]

test_start = pd.to_datetime(TEST_START)
test_end = pd.to_datetime(TEST_END)

def prepare_data(df, test_start, test_end, FEATURE_COLS, region=None):
    """
    Prepare train and test data for a given time period and region.
    """
    # Filter by region if specified
    if region is not None:
        df = df[df['region'] == region].copy()

    # Convert string dates to datetime
    test_start_dt = pd.Timestamp(test_start).tz_localize('America/New_York')
    test_end_dt = pd.Timestamp(test_end).tz_localize('America/New_York')

    # Split data
    test_data = df[(df['datetime'] >= test_start_dt) & (df['datetime'] <= test_end_dt)].copy()

    # Drop rows with missing lag features
    test_data = test_data.dropna(subset=FEATURE_COLS)

    # Prepare X and y
    X_test = test_data[FEATURE_COLS].values

    return X_test, test_data

# Load the saved models
regions= ['AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT', 'AP', 'BC', 'CE', 'DAY', 'DEOK',
                  'DOM', 'DPLCO', 'DUQ', 'EASTON', 'EKPC', 'JC', 'ME', 'OE', 'OVEC', 'PAPWR',
                  'PE', 'PEPCO', 'PLCO', 'PN', 'PS', 'RECO', 'SMECO', 'UGI', 'VMEU']
models_dir = os.path.join(OUTPUT_DIR, 'trained_models')
model_path = os.path.join(models_dir, 'hourly_load_models.pkl')

with open(model_path, 'rb') as f:
    hourly_best_models = pickle.load(f)

# Generate predictions for all regions
predictions_list = []

for region in regions:
    # Get model info for this region
    model_info = hourly_best_models[region]
    model = model_info['model']
    feature_cols = model_info['feature_cols']
    method = model_info['method']

    # Prepare prediction data
    pred_data = df[df['region'] == region].copy()

    if len(pred_data) == 0:
        continue
    # Get features
    X_pred = pred_data[feature_cols]

    y_pred = model.predict(X_pred)
    y_pred = np.round(y_pred).astype(int)
    # Store predictions with metadata
    pred_df = pred_data[['date']].copy()
    pred_df['region'] = region
    pred_df['predicted_load'] = y_pred
    pred_df['hour'] = pred_data['hour']
    predictions_list.append(pred_df)

# Combine all predictions
all_predictions = pd.concat(predictions_list, ignore_index=True)

# Create 3D structure: Pivot to get day x region x hour
predictions_pivot = all_predictions.pivot_table(
    index='date',
    columns=['region', 'hour'],
    values='predicted_load'
)

# Alternative: Create a proper 3D array
dates = sorted(all_predictions['date'].unique())
hours = list(range(24))

# Initialize 3D array: (days, regions, hours)
predictions_3d = np.zeros((len(dates), len(regions), len(hours)))

# Fill the 3D array
for i, date in enumerate(dates):
    for j, region in enumerate(regions):
        for k, hour in enumerate(hours):
            mask = (all_predictions['date'] == date) & \
                   (all_predictions['region'] == region) & \
                   (all_predictions['hour'] == hour)

            if mask.sum() > 0:
                predictions_3d[i, j, k] = all_predictions.loc[mask, 'predicted_load'].values[0]
            else:
                predictions_3d[i, j, k] = np.nan

all_predictions['predicted_load'] = all_predictions['predicted_load'].astype(int)
# Create a more accessible DataFrame format
predictions_df = all_predictions.pivot_table(
    index=['date', 'region'],
    columns='hour',
    values='predicted_load'
).reset_index()

# Rename hour columns for clarity
hour_cols = {i: f'hour_{i:02d}' for i in range(24)}
predictions_df = predictions_df.rename(columns=hour_cols)

# ======================================================================
# # Task 2
# ======================================================================

# Load the saved peak hour models
regions = ['AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT', 'AP', 'BC', 'CE', 'DAY', 'DEOK',
           'DOM', 'DPLCO', 'DUQ', 'EASTON', 'EKPC', 'JC', 'ME', 'OE', 'OVEC', 'PAPWR',
           'PE', 'PEPCO', 'PLCO', 'PN', 'PS', 'RECO', 'SMECO', 'UGI', 'VMEU']

models_dir = os.path.join(OUTPUT_DIR, 'trained_models')
model_path = os.path.join(models_dir, 'peak_hour_models.pkl')

with open(model_path, 'rb') as f:
    peak_hour_best_models = pickle.load(f)


# Dictionary to store peak hour predictions
peak_hour_predictions = {}

for region in regions:
    # Get model info for this region
    model_info = peak_hour_best_models[region]
    model = model_info['model']
    feature_cols = model_info['feature_cols']
    method = model_info['method']

    # Prepare prediction data for this region
    pred_data = df[df['region'] == region].copy()

    if len(pred_data) == 0:
        continue
    # Get features
    X_pred = pred_data[feature_cols]

    # Get predicted probabilities for being peak hour
    y_pred_proba = model.predict_proba(X_pred)[:, 1]  # Probability of class 1 (is_peak_hour=True)

    # Add predictions to dataframe
    pred_data['peak_hour_probability'] = y_pred_proba

    # For each date, find the hour with maximum probability
    region_peak_hours = {}
    for date in pred_data['date'].unique():
        date_data = pred_data[pred_data['date'] == date]
        # Find hour with maximum probability
        peak_hour_idx = date_data['peak_hour_probability'].idxmax()
        peak_hour = date_data.loc[peak_hour_idx, 'hour']
        region_peak_hours[date] = int(peak_hour)

    peak_hour_predictions[region] = region_peak_hours


# Get all unique dates
all_dates = sorted(df['date'].unique())

# Create DataFrame with regions as rows and dates as columns
peak_hour_df = pd.DataFrame(peak_hour_predictions).T
peak_hour_df = peak_hour_df.sort_index()  # Sort by region name

# Ensure columns are sorted by date
peak_hour_df = peak_hour_df[sorted(peak_hour_df.columns)]


# ======================================================================
# # Task 3
# ======================================================================

feature_cols = [
    # Temperature features
    'temperature_max', 'temperature_min', 'temperature_mean', 'temperature_std', 'temp_range',
    'hdd', 'cdd', 'is_very_hot', 'is_very_cold',
    # Other weather
    'humidity_max', 'humidity_min', 'humidity_mean',
    'precipitation_sum', 'precipitation_max',
    'wind_speed_max', 'wind_speed_mean',
    # Temporal features
    'month', 'day_of_week', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
]
FP_COST = 1  # False Positive: predict peak when not peak
FN_COST = 4  # False Negative: miss a peak day

# Get US holidays
us_holidays = holidays.US(years=range(2016, 2026))

df['date'] = df['datetime'].dt.date

# Aggregate to daily level - WEATHER ONLY (no load features except for labeling)
daily_agg = df.groupby(['date', 'region']).agg({
    'temperature': ['max', 'min', 'mean', 'std'],
    'humidity': ['max', 'min', 'mean'],
    'precipitation': ['sum', 'max'],
    'wind_speed': ['max', 'mean']
}).reset_index()

# Flatten column names
daily_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in daily_agg.columns.values]

# Convert date to datetime
daily_agg['date'] = pd.to_datetime(daily_agg['date'])

# Add temporal features
daily_agg['year'] = daily_agg['date'].dt.year
daily_agg['month'] = daily_agg['date'].dt.month
daily_agg['day_of_week'] = daily_agg['date'].dt.dayofweek
daily_agg['day_of_month'] = daily_agg['date'].dt.day
daily_agg['week_of_year'] = daily_agg['date'].dt.isocalendar().week
daily_agg['is_weekend'] = (daily_agg['day_of_week'] >= 5).astype(int)
daily_agg['is_holiday'] = daily_agg['date'].apply(lambda x: x in us_holidays).astype(int)

# Temperature features
daily_agg['temp_range'] = daily_agg['temperature_max'] - daily_agg['temperature_min']

# Heating/Cooling Degree Days (base 65°F)
daily_agg['hdd'] = np.maximum(0, 65 - daily_agg['temperature_mean'])
daily_agg['cdd'] = np.maximum(0, daily_agg['temperature_mean'] - 65)

# Extreme temperature indicators
daily_agg['is_very_hot'] = (daily_agg['temperature_max'] > 85).astype(int)
daily_agg['is_very_cold'] = (daily_agg['temperature_min'] < 32).astype(int)

# Cyclical encoding for temporal features
daily_agg['month_sin'] = np.sin(2 * np.pi * daily_agg['month'] / 12)
daily_agg['month_cos'] = np.cos(2 * np.pi * daily_agg['month'] / 12)
daily_agg['dow_sin'] = np.sin(2 * np.pi * daily_agg['day_of_week'] / 7)
daily_agg['dow_cos'] = np.cos(2 * np.pi * daily_agg['day_of_week'] / 7)

test_df = daily_agg[
        (daily_agg['date'] >= test_start) &
        (daily_agg['date'] <= test_end)
    ].copy()

# Load the saved peak days models
regions = ['AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT', 'AP', 'BC', 'CE', 'DAY', 'DEOK',
           'DOM', 'DPLCO', 'DUQ', 'EASTON', 'EKPC', 'JC', 'ME', 'OE', 'OVEC', 'PAPWR',
           'PE', 'PEPCO', 'PLCO', 'PN', 'PS', 'RECO', 'SMECO', 'UGI', 'VMEU']

models_dir = os.path.join(OUTPUT_DIR, 'trained_models')
model_path = os.path.join(models_dir, 'peak_days_models.pkl')

with open(model_path, 'rb') as f:
    peak_days_best_models = pickle.load(f)

# Dictionary to store peak days predictions
peak_days_predictions = {}

for region in tqdm(regions, desc="Predicting peak days"):
    # Get model info for this region
    model_info = peak_days_best_models[region]
    model = model_info['model']
    threshold = model_info['threshold']
    
    # Prepare prediction data for this region
    pred_data = test_df[test_df['region'] == region].copy()
    
    if len(pred_data) == 0:
        continue
    
    # Get features (same columns used in training)
    X_pred = pred_data[feature_cols]
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_pred)[:, 1]
    
    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Add predictions to dataframe
    pred_data['peak_day_pred'] = y_pred
    
    # Store predictions by date
    region_predictions = {}
    for idx, row in pred_data.iterrows():
        region_predictions[row['date']] = row['peak_day_pred']
    
    peak_days_predictions[region] = region_predictions


# Create DataFrame with regions as rows and dates as columns
peak_days_df = pd.DataFrame(peak_days_predictions).T
peak_days_df = peak_days_df.sort_index()  # Sort by region name

# Ensure columns are sorted by date
peak_days_df = peak_days_df[sorted(peak_days_df.columns)]


# ======================================================================
# ## Generate Tomorrow's Predictions and Save to CSV
# ======================================================================

# Get tomorrow's date
from datetime import datetime, timedelta
tomorrow = datetime.now() + timedelta(days=1)
tomorrow_date = tomorrow.date()
tomorrow_str = tomorrow.strftime('%Y-%m-%d')

# Read from existing prediction files
predictions_dir = os.path.join(OUTPUT_DIR, 'predictions')

# Build output in required format
output_values = [f'"{tomorrow_str}"']

# 1. HOURLY LOAD PREDICTIONS (L1_00 to L29_23) from predictions_df
for region in regions:
    region_data = predictions_df[(predictions_df['date'] == tomorrow_date) & 
                                  (predictions_df['region'] == region)]
    
    if len(region_data) > 0:
        hour_cols = [f'hour_{i:02d}' for i in range(24)]
        loads = region_data[hour_cols].values[0]
        # loads = [int(round(load)) for load in loads]
        output_values.extend([str(load) for load in loads])
    else:
        output_values.extend(['0'] * 24)

# 2. PEAK HOUR PREDICTIONS (PH_1 to PH_29) from peak_hour_df
for region in regions:
    if tomorrow_date in peak_hour_df.columns and region in peak_hour_df.index:
        peak_hour = int(peak_hour_df.loc[region, tomorrow_date])
        output_values.append(str(peak_hour))
    else:
        output_values.append('0')

# 3. PEAK DAY PREDICTIONS (PD_1 to PD_29) from peak_days_df
for region in regions:
    if tomorrow_date in peak_days_df.columns and region in peak_days_df.index:
        peak_day = int(peak_days_df.loc[region, tomorrow_date])
        output_values.append(str(peak_day))
    else:
        output_values.append('0')

# Create the output line
output_line = ','.join(output_values)

print(output_line)
predictions_dir = os.path.join(OUTPUT_DIR, 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

predictions_file = os.path.join(predictions_dir, 'predictions.csv')

# Append to file (or create if doesn't exist)
with open(predictions_file, 'a') as f:
    f.write(output_line + '\n')