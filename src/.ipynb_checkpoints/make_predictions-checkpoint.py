#!/usr/bin/env python3
"""
Load Forecasting Prediction Script
Generates predictions for tomorrow in the required CSV format.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import warnings
import pickle
import holidays

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

REGIONS = ['AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT', 'AP', 'BC', 'CE', 'DAY', 'DEOK',
           'DOM', 'DPLCO', 'DUQ', 'EASTON', 'EKPC', 'JC', 'ME', 'OE', 'OVEC', 'PAPWR',
           'PE', 'PEPCO', 'PLCO', 'PN', 'PS', 'RECO', 'SMECO', 'UGI', 'VMEU']

ZONE_COORDS = pd.DataFrame({
    'load_area': REGIONS,
    'lat': [39.45, 37.25, 38.45, 38.20, 39.90, 37.30, 40.80, 41.85, 39.75, 39.10,
            37.55, 38.90, 40.45, 39.55, 37.75, 40.35, 40.20, 41.10, 38.85, 40.70,
            40.00, 38.90, 40.95, 41.15, 40.75, 41.00, 38.40, 40.25, 37.30],
    'lon': [-74.50, -81.30, -81.60, -83.10, -82.90, -80.90, -79.95, -86.10, -84.20, -84.50,
            -77.45, -75.50, -79.90, -75.10, -84.30, -74.65, -76.00, -81.25, -82.85, -77.80,
            -75.20, -76.95, -77.40, -77.80, -74.15, -74.10, -76.70, -75.65, -76.00]
})

OUTPUT_DIR = 'output'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_models')

# ============================================================================
# WEATHER DATA FUNCTIONS
# ============================================================================

def fetch_weather_forecast(lat, lon, date, tries=3):
    """Fetch weather forecast data for a specific date and location"""
    api = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m"
    
    url = f"{api}?latitude={lat}&longitude={lon}&hourly={hourly_vars}&timezone=UTC&forecast_days=16"
    
    for attempt in range(1, tries + 1):
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
                    
                    # Filter to specific date
                    df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
                    date_str = date.strftime('%Y-%m-%d')
                    mask = (df['datetime_beginning_utc'] >= date_str) & \
                           (df['datetime_beginning_utc'] < (date + timedelta(days=1)).strftime('%Y-%m-%d'))
                    df = df[mask].copy()
                    
                    return df
        except Exception as e:
            if attempt == tries:
                raise Exception(f"Failed to fetch weather data: {e}")
    
    return None

def download_weather_for_date(date):
    """Download weather forecast for all zones for a specific date"""
    all_data = []
    
    for _, row in ZONE_COORDS.iterrows():
        zone = row['load_area']
        lat = row['lat']
        lon = row['lon']
        
        df = fetch_weather_forecast(lat, lon, date)
        
        if df is not None and not df.empty:
            df['load_area'] = zone
            all_data.append(df)
    
    if not all_data:
        raise Exception("Failed to download weather data for any zone")
    
    # Combine all zones
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert to Eastern Time
    utc = pytz.UTC
    eastern = pytz.timezone('US/Eastern')
    combined_df['datetime_beginning_ept'] = combined_df['datetime_beginning_utc'].dt.tz_localize(utc).dt.tz_convert(eastern)
    
    # Remove UTC column
    combined_df = combined_df.drop('datetime_beginning_utc', axis=1)
    
    # Reorder columns
    column_order = ['datetime_beginning_ept', 'load_area', 'temp', 'humidity', 'dew_point', 'precip', 'wind']
    combined_df = combined_df[column_order]
    
    return combined_df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(weather_df):
    """Create features from weather data"""
    df = weather_df.copy()
    
    # Rename load_area to region for consistency
    df['region'] = df['load_area']
    
    # Extract time features
    df['date'] = pd.to_datetime(df['datetime_beginning_ept']).dt.date
    df['hour'] = pd.to_datetime(df['datetime_beginning_ept']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['datetime_beginning_ept']).dt.dayofweek
    df['month'] = pd.to_datetime(df['datetime_beginning_ept']).dt.month
    df['day_of_year'] = pd.to_datetime(df['datetime_beginning_ept']).dt.dayofyear
    df['week_of_year'] = pd.to_datetime(df['datetime_beginning_ept']).dt.isocalendar().week
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Holiday indicator (US holidays)
    us_holidays = holidays.US(years=df['date'].apply(lambda x: x.year).unique().tolist())
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays).astype(int)
    
    # Create daily aggregates
    daily_agg = df.groupby(['region', 'date']).agg({
        'temp': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max'],
        'dew_point': ['mean', 'min', 'max'],
        'precip': ['sum', 'max'],
        'wind': ['mean', 'max'],
        'day_of_week': 'first',
        'month': 'first',
        'day_of_year': 'first',
        'week_of_year': 'first',
        'is_weekend': 'first',
        'is_holiday': 'first'
    }).reset_index()
    
    # Flatten column names
    daily_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in daily_agg.columns.values]
    
    # Rename columns to match training
    daily_agg = daily_agg.rename(columns={
        'temp_mean': 'temp',
        'temp_min': 'temp_min',
        'temp_max': 'temp_max',
        'temp_std': 'temp_std',
        'humidity_mean': 'humidity',
        'humidity_min': 'humidity_min',
        'humidity_max': 'humidity_max',
        'dew_point_mean': 'dew_point',
        'dew_point_min': 'dew_point_min',
        'dew_point_max': 'dew_point_max',
        'precip_sum': 'precip',
        'precip_max': 'precip_max',
        'wind_mean': 'wind',
        'wind_max': 'wind_max'
    })
    
    return df, daily_agg

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def load_models():
    """Load all trained models"""
    with open(os.path.join(MODELS_DIR, 'hourly_load_models.pkl'), 'rb') as f:
        hourly_models = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'peak_hour_models.pkl'), 'rb') as f:
        peak_hour_models = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'peak_days_models.pkl'), 'rb') as f:
        peak_days_models = pickle.load(f)
    
    return hourly_models, peak_hour_models, peak_days_models

def predict_hourly_loads(hourly_df, hourly_models):
    """Predict hourly loads for all regions"""
    predictions = {}
    
    for region in REGIONS:
        region_data = hourly_df[hourly_df['region'] == region].sort_values('hour').copy()
        
        if len(region_data) == 0:
            predictions[region] = [0] * 24
            continue
        
        model_info = hourly_models[region]
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        X_pred = region_data[feature_cols]
        y_pred = np.round(model.predict(X_pred)).astype(int)
        
        predictions[region] = y_pred.tolist()
    
    return predictions

def predict_peak_hours(hourly_df, peak_hour_models):
    """Predict peak hour for each region"""
    predictions = {}
    
    for region in REGIONS:
        region_data = hourly_df[hourly_df['region'] == region].copy()
        
        if len(region_data) == 0:
            predictions[region] = 0
            continue
        
        model_info = peak_hour_models[region]
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        X_pred = region_data[feature_cols]
        y_pred_proba = model.predict_proba(X_pred)[:, 1]
        
        # Find hour with maximum probability
        peak_hour = int(region_data.iloc[y_pred_proba.argmax()]['hour'])
        predictions[region] = peak_hour
    
    return predictions

def predict_peak_days(daily_df, peak_days_models):
    """Predict whether each region will have a peak day"""
    predictions = {}
    
    for region in REGIONS:
        region_data = daily_df[daily_df['region'] == region].copy()
        
        if len(region_data) == 0:
            predictions[region] = 0
            continue
        
        model_info = peak_days_models[region]
        model = model_info['model']
        threshold = model_info.get('threshold', 0.5)
        
        # Get feature columns - handle different possible storage formats
        if 'feature_cols' in model_info:
            feature_cols = model_info['feature_cols']
        else:
            # Use common features if not specified
            feature_cols = ['temp', 'temp_min', 'temp_max', 'temp_std',
                          'humidity', 'humidity_min', 'humidity_max',
                          'dew_point', 'dew_point_min', 'dew_point_max',
                          'precip', 'precip_max', 'wind', 'wind_max',
                          'day_of_week', 'month', 'day_of_year', 'week_of_year',
                          'is_weekend', 'is_holiday']
            feature_cols = [col for col in feature_cols if col in region_data.columns]
        
        X_pred = region_data[feature_cols].iloc[0:1]
        y_pred_proba = model.predict_proba(X_pred)[0, 1]
        
        peak_day = 1 if y_pred_proba >= threshold else 0
        predictions[region] = peak_day
    
    return predictions

# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def make_predictions():
    """Generate predictions for tomorrow and output in required CSV format"""
    
    # Get tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    prediction_date = tomorrow.strftime('%Y-%m-%d')
    
    # Download weather forecast
    weather_df = download_weather_for_date(tomorrow)
    
    # Create features
    hourly_df, daily_df = create_features(weather_df)
    
    # Load models
    hourly_models, peak_hour_models, peak_days_models = load_models()
    
    # Make predictions
    hourly_loads = predict_hourly_loads(hourly_df, hourly_models)
    peak_hours = predict_peak_hours(hourly_df, peak_hour_models)
    peak_days = predict_peak_days(daily_df, peak_days_models)
    
    # Build output in required format
    output = [f'"{prediction_date}"']
    
    # Add hourly loads (L1_00 to L29_23)
    for region in REGIONS:
        output.extend([str(load) for load in hourly_loads[region]])
    
    # Add peak hours (PH_1 to PH_29)
    for region in REGIONS:
        output.append(str(peak_hours[region]))
    
    # Add peak days (PD_1 to PD_29)
    for region in REGIONS:
        output.append(str(peak_days[region]))
    
    # Print the CSV line
    print(','.join(output))

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        make_predictions()
    except Exception as e:
        import sys
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
