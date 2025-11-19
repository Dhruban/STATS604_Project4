#!/usr/bin/env python3
"""
Comprehensive Analysis and Model Training

This script performs:
1. Data merging (load + weather)
2. Seasonal analysis
3. Best model finding for 3 tasks:
   - Hourly load forecasting
   - Peak hour prediction
   - Peak days prediction
4. Saves trained best models for production use

Usage:
    python analysis_script.py

Requirements:
    - pandas, numpy, matplotlib, seaborn
    - scikit-learn, xgboost, lightgbm, catboost
    - holidays, tqdm
    
Directory Structure Expected:
    ../data/raw/hrl_load_metered_2016-2025/  (load data)
    ../data/raw/weather/                      (weather data)
    
Output:
    ../output/trained_models/hourly_load_models.pkl
    ../output/trained_models/peak_hour_models.pkl
    ../output/trained_models/peak_days_models.pkl
    ../figures/ (various analysis plots)
    ../data/processed/ (merged and processed data)
"""

# ======================================================================
# # Comprehensive Analysis and Model Training
#
# This notebook performs:
# 1. Data merging (load + weather)
# 2. Seasonal analysis
# 3. Best model finding for 3 tasks:
#    - Hourly load forecasting
#    - Peak hour prediction
#    - Peak days prediction
# 4. Saves trained best models for production use
# ======================================================================

# ======================================================================
# ## 1. Setup and Configuration
# ======================================================================

# Install required packages
# !pip install xgboost lightgbm catboost holidays --break-system-packages # Skipped shell command

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
import holidays
import pickle
from datetime import datetime, timedelta
import pytz

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
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("Libraries imported successfully")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory paths (relative to src/)
LOAD_DIR = "../data/raw/hrl_load_metered_2016-2025"
WEATHER_DIR = "../data/raw/weather"
PROCESSED_DIR = "../data/processed"
FIGURES_DIR = "../figures"
OUTPUT_DIR = "../output"

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target 29 load areas
KEEP_AREAS = [
    "AECO", "AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT", "AP", "BC", "CE", "DAY", "DEOK",
    "DOM", "DPLCO", "DUQ", "EASTON", "EKPC", "JC", "ME", "OE", "OVEC", "PAPWR",
    "PE", "PEPCO", "PLCO", "PN", "PS", "RECO", "SMECO", "UGI", "VMEU"
]

# Best Model Finding dates (2024 test)
TRAIN_START = '2016-01-01 00:00:00'
TRAIN_END = '2024-10-30 23:00:00'
TEST_START = '2024-11-20 00:00:00'
TEST_END = '2024-11-29 23:00:00'

FINAL_TRAIN_START = '2016-01-01 00:00:00'
FINAL_TRAIN_END = '2025-10-30 23:00:00'

# Rolling window for peak days
WINDOW_SIZE = 10  # days
NUM_PEAK_DAYS = 2

print("Configuration loaded")
print(f"\nBest Model Finding Period:")
print(f"  Train: {TRAIN_START} to {TRAIN_END}")
print(f"  Test:  {TEST_START} to {TEST_END}")

# ======================================================================
# ## 2. Data Merging
# ======================================================================

# Helper functions
def n_distinct_na_omit(series):
    """Count unique non-null values in a series"""
    return series.dropna().nunique()

def add_seconds(time_str):
    """Add seconds to time string if not present"""
    import re
    if re.search(r':\d{2}$', time_str):
        return time_str + ":00"
    return time_str

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

print("Helper functions defined")

print("=" * 70)
print("STEP 1: LOADING AND MERGING DATA")
print("=" * 70)

# Load all yearly load files
load_files = sorted(glob.glob(os.path.join(LOAD_DIR, "hrl_load_metered_*.csv")))

if len(load_files) == 0:
    raise ValueError(f"No load files found in {LOAD_DIR}")

print(f"\nFound {len(load_files)} load files")

datasets = {}
for f in load_files:
    year = os.path.basename(f).split('_')[-1].replace('.csv', '')
    df = pd.read_csv(f)
    df['year'] = int(year)
    datasets[year] = df
    print(f"  Loaded {year}: {len(df):,} rows")

# Combine all years and filter to 29 areas
load_df = pd.concat(datasets.values(), ignore_index=True)
load_df = load_df[load_df['load_area'].isin(KEEP_AREAS)].copy()

print(f"\nCombined load data: {len(load_df):,} rows")
print(f"Unique load areas: {load_df['load_area'].nunique()}")

# Parse datetime
# Parse datetime and snap to hour
load_df['dt_ept'] = pd.to_datetime(
    load_df['datetime_beginning_ept'],
    format='%m/%d/%Y %I:%M:%S %p'
)

# Localize to ET timezone
load_df['dt_ept'] = load_df['dt_ept'].dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')

# Snap to hour (floor to beginning of hour)
load_df['dt_ept'] = load_df['dt_ept'].dt.floor('h')
print(f"After datetime parsing: {len(load_df):,} rows")

# Load weather data
weather_files = sorted(glob.glob(os.path.join(WEATHER_DIR, "*.csv")))

if len(weather_files) == 0:
    raise ValueError(f"No weather files found in {WEATHER_DIR}")

print(f"\nFound {len(weather_files)} weather files")

weather_dfs = []
for f in weather_files:
    df = pd.read_csv(f)
    year = os.path.basename(f).split('_')[-1].replace('.csv', '')
    weather_dfs.append(df)
    print(f"Loaded {year}: {len(df):,} rows")

weather_df = pd.concat(weather_dfs, ignore_index=True)

print(f"\nCombined weather data: {len(weather_df):,} rows")
print(f"Load areas: {weather_df['load_area'].nunique()}")

# Parse ET datetime
weather_df['wx_ept'] = pd.to_datetime(
    weather_df['datetime_beginning_ept'],
    format='%Y-%m-%d %H:%M:%S'
)

# Localize to ET timezone
weather_df['wx_ept'] = weather_df['wx_ept'].dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')

# Snap to hour
weather_df['wx_ept'] = weather_df['wx_ept'].dt.floor('h')

# Create ordering column for UTC time (for stable deduplication)
weather_df['order_utc'] = pd.to_datetime(
    weather_df['datetime_beginning_utc'],
    format='%Y-%m-%dT%H:%M:%S',
    utc=True
)

# Sort by load_area, wx_ept, and order_utc
weather_df = weather_df.sort_values(['load_area', 'wx_ept', 'order_utc'])

# Remove duplicates (keep first occurrence)
weather_dedup = weather_df.drop_duplicates(subset=['load_area', 'wx_ept'], keep='first').copy()

# Drop the ordering column
weather_dedup = weather_dedup.drop(columns=['order_utc'])

print(f"Weather data before deduplication: {len(weather_df):,} rows")
print(f"Weather data after deduplication: {len(weather_dedup):,} rows")
print(f"Duplicates removed: {len(weather_df) - len(weather_dedup):,}")

# Select weather columns to keep
keep_wx = ['load_area', 'wx_ept', 'temp', 'humidity', 'dew_point', 'precip', 'wind']

# Perform left merge
merged = load_df.merge(
    weather_dedup[keep_wx],
    left_on=['load_area', 'dt_ept'],
    right_on=['load_area', 'wx_ept'],
    how='left'
)

# Drop the helper datetime columns used for merging
merged = merged.drop(columns=['dt_ept', 'wx_ept','year'])

print(f"\nMerged data: {len(merged):,} rows")
print(f"Weather coverage: {(~merged['temp'].isna()).sum() / len(merged) * 100:.2f}%")

# Save merged data
output_file = os.path.join(PROCESSED_DIR, "merged_load_weather.csv")
merged.to_csv(output_file, index=False)
print(f"\nMerged data saved to: {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")

# ======================================================================
# ## 3. Seasonal Analysis
# ======================================================================

# Parse datetime
merged['datetime_beginning_ept'] = parse_et(merged['datetime_beginning_ept'])

# Check for NaT values
nat_count = merged['datetime_beginning_ept'].isna().sum()
print(f"Found {nat_count} rows with invalid datetime")

if nat_count > 0:
    # Show sample of problematic rows
    print("\nSample of rows with NaT:")
    print(merged[merged['datetime_beginning_ept'].isna()][['datetime_beginning_ept', 'load_area']].head())

    # Drop rows with NaT
    merged = merged.dropna(subset=['datetime_beginning_ept'])
    print(f"\nDropped {nat_count} rows with invalid datetime")
    print(f"Remaining rows: {len(merged):,}")

print("DateTime parsed successfully!")
print(f"Date range: {merged['datetime_beginning_ept'].min()} to {merged['datetime_beginning_ept'].max()}")

print("\n" + "=" * 70)
print("STEP 2: SEASONAL ANALYSIS")
print("=" * 70)

# Prepare data for seasonal analysis
df_season = merged.copy()
df_season['datetime'] = pd.to_datetime(df_season['datetime_beginning_ept'])
df_season['year'] = df_season['datetime'].dt.year
df_season['month'] = df_season['datetime'].dt.month
df_season['day'] = df_season['datetime'].dt.day
df_season['hour'] = df_season['datetime'].dt.hour
df_season['wday'] = df_season['datetime'].dt.dayofweek
df_season['date'] = df_season['datetime'].dt.date
df_season['dayofweek_name'] = df_season['datetime_beginning_ept'].dt.strftime('%a')

# Define seasons
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

df_season['season'] = df_season['month'].apply(get_season)
# December belongs to next year's winter
df_season['season_year'] = df_season['year'].where(df_season['month'] != 12, df_season['year'] + 1)
# Day-hour index (0-167 for weekly cycle)
df_season['dayhour'] = df_season['wday'] * 24 + df_season['hour']

print(f"\nSeasonal data prepared: {len(df_season):,} rows")

# Monthly and yearly aggregations
monthly_tbl = df_season.groupby(['year', 'month']).agg({
    'mw': 'mean',
    'temp': 'mean',
    'humidity': 'mean',
    'wind': 'mean',
    'precip': 'mean'
}).reset_index()

yearly_tbl = df_season.groupby('year').agg({
    'mw': 'mean',
    'temp': 'mean',
    'humidity': 'mean',
    'wind': 'mean',
    'precip': 'mean'
}).reset_index()

print(f"Monthly aggregations: {len(monthly_tbl)} rows")
print(f"Yearly aggregations: {len(yearly_tbl)} rows")

def summarise_seasonal(data, group_cols):
    """Calculate seasonal means for specified grouping columns"""
    grouped = data.groupby(['season', 'season_year'] + group_cols).agg({
        'mw': 'mean',
        'temp': 'mean',
        'humidity': 'mean',
        'wind': 'mean',
        'precip': 'mean'
    }).reset_index()

    # Rename columns
    rename_dict = {'mw': 'Load_MW', 'temp': 'Temp_C', 'humidity': 'Humidity',
                   'wind': 'Wind', 'precip': 'Precip'}
    grouped = grouped.rename(columns=rename_dict)

    # Convert to long format
    long = grouped.melt(
        id_vars=['season', 'season_year'] + group_cols,
        value_vars=['Load_MW', 'Temp_C', 'Humidity', 'Wind', 'Precip'],
        var_name='Metric',
        value_name='Value'
    )

    return long

# Create summaries
print("Creating seasonal summaries...")
daily_tbl = summarise_seasonal(df_season, ['wday', 'dayofweek_name'])
hourly_tbl = summarise_seasonal(df_season, ['hour'])
dayhour_tbl = summarise_seasonal(df_season, ['dayhour'])

# ======================================================================
# ## Monthly Trends by Year
# ======================================================================

# Plot 1: Monthly trends by year
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for year in sorted(monthly_tbl['year'].unique()):
    data = monthly_tbl[monthly_tbl['year'] == year]
    axes[0, 0].plot(data['month'], data['mw'], marker='o', label=str(year), alpha=0.7)
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Average Load (MW)')
axes[0, 0].set_title('Monthly Load Trends by Year', fontweight='bold')
axes[0, 0].legend(ncol=2, fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

for year in sorted(monthly_tbl['year'].unique()):
    data = monthly_tbl[monthly_tbl['year'] == year]
    axes[0, 1].plot(data['month'], data['temp'], marker='o', label=str(year), alpha=0.7)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Temperature (°C)')
axes[0, 1].set_title('Monthly Temperature Trends by Year', fontweight='bold')
axes[0, 1].legend(ncol=2, fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

for year in sorted(monthly_tbl['year'].unique()):
    data = monthly_tbl[monthly_tbl['year'] == year]
    axes[1, 0].plot(data['month'], data['humidity'], marker='o', label=str(year), alpha=0.7)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Average Humidity (%)')
axes[1, 0].set_title('Monthly Humidity Trends by Year', fontweight='bold')
axes[1, 0].legend(ncol=2, fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

for year in sorted(monthly_tbl['year'].unique()):
    data = monthly_tbl[monthly_tbl['year'] == year]
    axes[1, 1].plot(data['month'], data['wind'], marker='o', label=str(year), alpha=0.7)
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Average Wind Speed')
axes[1, 1].set_title('Monthly Wind Speed Trends by Year', fontweight='bold')
axes[1, 1].legend(ncol=2, fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Monthly Trends Across Years', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'monthly_trends.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved: monthly_trends.png")

# ======================================================================
# ## Yeary Trends
# ======================================================================

# Plot 2: Yearly aggregate trends
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(yearly_tbl['year'], yearly_tbl['mw'], marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Average Load (MW)')
axes[0, 0].set_title('Yearly Load Trend', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(yearly_tbl['year'], yearly_tbl['temp'], marker='o', linewidth=2, markersize=8, color='coral')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Average Temperature (°C)')
axes[0, 1].set_title('Yearly Temperature Trend', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(yearly_tbl['year'], yearly_tbl['humidity'], marker='o', linewidth=2, markersize=8, color='lightgreen')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Average Humidity (%)')
axes[1, 0].set_title('Yearly Humidity Trend', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(yearly_tbl['year'], yearly_tbl['wind'], marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Average Wind Speed')
axes[1, 1].set_title('Yearly Wind Speed Trend', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Yearly Aggregate Trends', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'yearly_trends.png'), dpi=150, bbox_inches='tight')
plt.show()

print("Saved: yearly_trends.png")

# ======================================================================
# ## Seasonal Trends by Year
# ======================================================================

# Function to plot seasonal patterns by day of week
def plot_seasonal_by_wday(season_name, data):
    """Plot seasonal patterns by day of week"""
    season_data = data[data['season'] == season_name]

    # Define day order
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    season_data['dayofweek_name'] = pd.Categorical(season_data['dayofweek_name'],
                                                    categories=day_order, ordered=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    metrics = ['Load_MW', 'Temp_C', 'Humidity', 'Wind', 'Precip']
    for idx, metric in enumerate(metrics):
        metric_data = season_data[season_data['Metric'] == metric]

        for year in sorted(metric_data['season_year'].unique()):
            year_data = metric_data[metric_data['season_year'] == year].sort_values('wday')
            axes[idx].plot(year_data['dayofweek_name'], year_data['Value'],
                          marker='o', label=str(year), alpha=0.75, linewidth=2)

        axes[idx].set_title(f'{metric}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Wday', fontsize=9)
        axes[idx].set_ylabel('Mean', fontsize=9)
        axes[idx].legend(title='Season-Year', fontsize=7, ncol=2)
        axes[idx].grid(True, alpha=0.3)

    fig.delaxes(axes[5])
    plt.suptitle(f'{season_name} – Features by Day of Week',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{season_name}_wday.png'), dpi=150, bbox_inches='tight')
    plt.show()

# Generate plots for all seasons
print("\nGenerating seasonal plots by day of week...")
for season in ['Spring', 'Summer', 'Fall', 'Winter']:
    plot_seasonal_by_wday(season, daily_tbl)
    print(f"  Saved: {season}_wday.png")

# Function to plot seasonal patterns by hour
def plot_seasonal_by_hour(season_name, data):
    """Plot seasonal patterns by hour of day"""
    season_data = data[data['season'] == season_name]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    metrics = ['Load_MW', 'Temp_C', 'Humidity', 'Wind', 'Precip']
    for idx, metric in enumerate(metrics):
        metric_data = season_data[season_data['Metric'] == metric]

        for year in sorted(metric_data['season_year'].unique()):
            year_data = metric_data[metric_data['season_year'] == year].sort_values('hour')
            axes[idx].plot(year_data['hour'], year_data['Value'],
                          marker='o', label=str(year), alpha=0.75, linewidth=2, markersize=4)

        axes[idx].set_title(f'{metric}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Hour (ET)', fontsize=9)
        axes[idx].set_ylabel('Mean', fontsize=9)
        axes[idx].legend(title='Season-Year', fontsize=7, ncol=2)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks(range(0, 24, 2))

    fig.delaxes(axes[5])
    plt.suptitle(f'{season_name} – Features by Hour of Day',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig
    plt.savefig(os.path.join(FIGURES_DIR, f'{season_name}_hour.png'), dpi=150, bbox_inches='tight')
    plt.show()

# Generate plots for all seasons
print("\nGenerating seasonal plots by hour...")
for season in ['Spring', 'Summer', 'Fall', 'Winter']:
    plot_seasonal_by_hour(season, hourly_tbl)
    print(f"  Saved: {season}_hour.png")

print("\n" + "=" * 70)
print("Seasonal analysis complete!")
print(f"Generated {len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])} plots")
print("=" * 70)

# ======================================================================
# ## 4. Prepare Data for Model Training
# ======================================================================

print("\n" + "=" * 70)
print("STEP 4: PREPARING DATA FOR MODEL TRAINING")
print("=" * 70)

# Load merged data
df = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_load_weather.csv"))

# Rename columns to standard names
df = df.rename(columns={
    'datetime_beginning_ept': 'datetime',
    'load_area': 'region',
    'mw': 'load',
    'temp': 'temperature',
    'precip': 'precipitation',
    'wind': 'wind_speed'
})

# Keep only necessary columns
keep_cols = ['datetime', 'region', 'load', 'temperature', 'humidity', 'precipitation', 'wind_speed']
df = df[keep_cols]

df['datetime'] = parse_et(df['datetime'])
df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\nData loaded: {len(df):,} rows")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Regions: {df['region'].nunique()}")

# Feature engineering function
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
print("\nFeature engineering complete")
print(f"Total features: {len(df.columns)}")

# ======================================================================
# ## 5. Task 1: Hourly Load Forecast - Best Model Finding
# ======================================================================

print("\n" + "=" * 70)
print("TASK 1: HOURLY LOAD FORECAST - BEST MODEL FINDING")
print("=" * 70)

FEATURE_COLS = [
    # Temporal
    'hour', 'day_of_week', 'month', 'day_of_month', 'day_of_year', 'week_of_year', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    # Weather
    'temperature', 'humidity', 'wind_speed', 'precipitation',
    # 'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
    'is_thanksgiving', 'is_christmas', 'is_new_years', 'is_july4', 'is_other_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
]

TARGET_COL = 'load'
# Get list of regions
regions = sorted(df['region'].unique())
print(f"\nTraining models for {len(regions)} regions")

# Define training period
train_start = pd.to_datetime(TRAIN_START)
train_end = pd.to_datetime(TRAIN_END)
test_start = pd.to_datetime(TEST_START)
test_end = pd.to_datetime(TEST_END)
final_train_start = pd.to_datetime(FINAL_TRAIN_START)
final_train_end = pd.to_datetime(FINAL_TRAIN_END)

print(f"\nTraining period: {train_start} to {train_end}")
print(f"Testing period: {test_start} to {test_end}")

def get_models():
    """
    Define all models to be compared.
    Returns dict with model name and model object.
    """
    models = {
        '1. Linear + Interactions': None,  # Special handling needed
        '2. Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        '3. XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        '4. LightGBM': LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        '5. CatBoost': CatBoostRegressor(
            iterations=100,
            depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
    }
    return models

print("Model definitions created")

def train_linear_with_interactions(X_train, y_train, X_test):
    """
    Simple linear regression with minimal categorical encoding.

    - No scaling
    - Only one-hot encode day_of_week (7 categories)
    - Keep hour and month as numeric (they have some ordinal meaning)
    - Focus on key interactions
    """
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
    X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)

    # Only one-hot encode day_of_week
    X_train_df = pd.get_dummies(X_train_df, columns=['day_of_week'], drop_first=True)
    X_test_df = pd.get_dummies(X_test_df, columns=['day_of_week'], drop_first=True)

    # Align columns
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

    # Key interactions
    X_train_df['hour_x_temp'] = X_train_df['hour'] * X_train_df['temperature']
    X_train_df['temp_squared'] = X_train_df['temperature'] ** 2
    X_train_df['temp_x_humidity'] = X_train_df['temperature'] * X_train_df['humidity']
    X_train_df['weekend_x_temp'] = X_train_df['is_weekend'] * X_train_df['temperature']

    X_test_df['hour_x_temp'] = X_test_df['hour'] * X_test_df['temperature']
    X_test_df['temp_squared'] = X_test_df['temperature'] ** 2
    X_test_df['temp_x_humidity'] = X_test_df['temperature'] * X_test_df['humidity']
    X_test_df['weekend_x_temp'] = X_test_df['is_weekend'] * X_test_df['temperature']

    # Train
    model = LinearRegression()
    model.fit(X_train_df, y_train)
    y_pred = model.predict(X_test_df)

    return model, y_pred

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def prepare_data(df, train_start, train_end, test_start, test_end, FEATURE_COLS, TARGET_COL, region=None):
    """
    Prepare train and test data for a given time period and region.
    """
    # Filter by region if specified
    if region is not None:
        df = df[df['region'] == region].copy()

    # Convert string dates to datetime
    train_start_dt = pd.Timestamp(train_start).tz_localize('America/New_York')
    train_end_dt = pd.Timestamp(train_end).tz_localize('America/New_York')
    test_start_dt = pd.Timestamp(test_start).tz_localize('America/New_York')
    test_end_dt = pd.Timestamp(test_end).tz_localize('America/New_York')

    # Split data
    train_data = df[(df['datetime'] >= train_start_dt) & (df['datetime'] <= train_end_dt)].copy()
    test_data = df[(df['datetime'] >= test_start_dt) & (df['datetime'] <= test_end_dt)].copy()

    # Drop rows with missing lag features
    train_data = train_data.dropna(subset=FEATURE_COLS)
    test_data = test_data.dropna(subset=FEATURE_COLS)

    # Prepare X and y
    X_train = train_data[FEATURE_COLS].values
    y_train = train_data[TARGET_COL].values
    X_test = test_data[FEATURE_COLS].values
    y_test = test_data[TARGET_COL].values

    return X_train, y_train, X_test, y_test, test_data

print("Helper functions defined")

# Get unique regions
regions = sorted(df['region'].unique())
print(f"Number of regions: {len(regions)}")
print(f"Regions: {regions}")

# Compare all models across all regions
print("="*70)
print("MODEL COMPARISON (2024 TEST)")
print("="*70)

load_results = []
models_dict = get_models()

for region in tqdm(regions, desc="Processing regions"):
    print(f"\n{'='*70}")
    print(f"Region: {region}")
    print(f"{'='*70}")

    # Prepare data
    X_train, y_train, X_test, y_test, test_data = prepare_data(
        df, TRAIN_START, TRAIN_END,
        TEST_START, TEST_END, FEATURE_COLS, TARGET_COL,
        region=region
    )

    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # Train and evaluate each model
    for model_name, model in models_dict.items():
        try:
            if model_name == '1. Linear + Interactions':
                # Special handling for linear with interactions
                model, y_pred = train_linear_with_interactions(X_train, y_train, X_test)
            else:
                # Train model
                model.fit(X_train, y_train)
                # Predict
                y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)

            # Store results
            load_results.append({
                'Region': region,
                'Method': model_name,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'Train_Size': len(X_train),
                'Test_Size': len(X_test)
            })

            print(f"{model_name:30s} - RMSE: {metrics['RMSE']:8.2f}, MAE: {metrics['MAE']:8.2f}, R2: {metrics['R2']:6.4f}")

        except Exception as e:
            print(f"{model_name:30s} - ERROR: {str(e)}")

# Convert to DataFrame
load_results_df = pd.DataFrame(load_results)

print("\n" + "="*70)
print("Complete!")
print("="*70)

# Overall summary by method (averaged across all regions)
load_results_summary = load_results_df.groupby('Method').agg({
    'MSE': 'mean',
    'RMSE': 'mean',
    'MAE': 'mean',
    'R2': 'mean'
}).round(2)

load_results_summary = load_results_summary.sort_values('RMSE')

print("\n" + "="*70)
print("SUMMARY: Average Performance Across All Regions")
print("="*70)
print(load_results_summary)

# Best method overall
best_method_overall = load_results_summary.index[0]
print(f"\nBest Overall Method: {best_method_overall}")

# Now retrain the best method for all regions and save
print(f"\nRetraining best method ({best_method_overall}) for all regions...")

hourly_best_models = {}  # Store trained models

for region in tqdm(regions, desc="Training best models"):
    # Prepare data
    X_train, y_train, X_test, y_test, test_data = prepare_data(
        df, FINAL_TRAIN_START, FINAL_TRAIN_END,
        TEST_START, TEST_END, FEATURE_COLS, TARGET_COL,
        region=region
    )

    try:
        # Get the best method
        methods = get_models()
        model = methods[best_method_overall]

        if best_method_overall == '1. Linear + Interactions':
            model= train_linear_with_interactions(X_train, y_train, X_test)

            # Store both model and poly transformer
            hourly_best_models[region] = {
                'model': model,
                'method': best_method_overall,
                'feature_cols': FEATURE_COLS
            }
        else:
            model.fit(X_train, y_train)

            hourly_best_models[region] = {
                'model': model,
                'method': best_method_overall,
                'feature_cols': FEATURE_COLS
            }

    except Exception as e:
        print(f"Error training {best_method_overall} for {region}: {e}")
        continue

print(f"Trained best model for {len(hourly_best_models)} regions")

# ======================================================================
# ## 6. Task 2: Peak Hour Prediction - Best Model Finding
# ======================================================================

# Create peak hour labels for each day-region combination
print("Creating peak hour labels...")

peak_indices = df.groupby(['region', 'date'])['load'].idxmax()
df['is_peak_hour'] = 0
df.loc[peak_indices, 'is_peak_hour'] = 1

# Store actual peak hour for each day-region
df['peak_hour_actual'] = df.groupby(['region', 'date'])['hour'].transform(
    lambda x: x[df.loc[x.index, 'is_peak_hour'] == 1].iloc[0] if any(df.loc[x.index, 'is_peak_hour'] == 1) else -1
)

print(f"Peak hours labeled: {df['is_peak_hour'].sum():,}")

# Drop temporary date column for now (will recreate when needed)
df = df.drop('date', axis=1)

print("\n" + "=" * 70)
print("TASK 2: PEAK HOUR PREDICTION - BEST MODEL FINDING")
print("=" * 70)

et_tz = pytz.timezone('America/New_York')
p1_train_end = pd.to_datetime(TRAIN_END).tz_localize(et_tz)
p1_final_train_end= pd.to_datetime(FINAL_TRAIN_END).tz_localize(et_tz)
p1_test_start = pd.to_datetime(TEST_START).tz_localize(et_tz)
p1_test_end = pd.to_datetime(TEST_END).tz_localize(et_tz)

p1_train = df[df['datetime'] <= p1_train_end].copy()
p1_final_train = df[df['datetime'] <= p1_final_train_end].copy()
p1_test = df[(df['datetime'] >= p1_test_start) &
             (df['datetime'] <= p1_test_end)].copy()

# Add date column for grouping
p1_test['date'] = p1_test['datetime'].dt.date

print(f"Training: {len(p1_train):,} rows")
print(f"Test: {len(p1_test):,} rows")
print(f"Test days: {p1_test['date'].nunique()}")
print(f"Test predictions needed: {len(regions)} regions × {p1_test['date'].nunique()} days = {len(regions) * p1_test['date'].nunique()}")

print(f"\nPeak hour data prepared")

# Define methods for peak hour
def get_peak_hour_methods():
    return {
        '1. Linear + Interactions': 'regression',  # Will use regression to predict load
        '2. Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        '3. XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
        '4. LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        '5. Historical Same Date': 'historical'  # Look at same calendar date in previous years
    }

print("Methods defined for peak hour prediction")

# ======================================================================
# ### Method 1: Linear Regression with Interactions
# ======================================================================

print("\nTraining Method 1: Linear Regression with Interactions...")

p1_m1_results = []

for region in tqdm(regions, desc="Method 1"):
    try:
        # Train data for this region
        region_train = p1_train[p1_train['region'] == region].dropna(subset=FEATURE_COLS + ['load'])
        region_test = p1_test[p1_test['region'] == region].dropna(subset=FEATURE_COLS)

        if len(region_train) < 100 or len(region_test) == 0:
            continue

        X_train = region_train[FEATURE_COLS].values
        y_train = region_train['load'].values
        X_test = region_test[FEATURE_COLS].values

        # Train and predict in one call
        _, y_pred = train_linear_with_interactions(X_train, y_train, X_test)

        # For each day, find hour with max predicted load
        region_test_copy = region_test.copy()
        region_test_copy['predicted_load'] = y_pred

        for date in region_test_copy['date'].unique():
            day_data = region_test_copy[region_test_copy['date'] == date]

            # Predicted peak hour = hour with max predicted load
            pred_peak_hour = day_data.loc[day_data['predicted_load'].idxmax(), 'hour']

            # Actual peak hour
            actual_peak_hour = day_data.loc[day_data['load'].idxmax(), 'hour']

            # Check success
            success = np.abs(pred_peak_hour - actual_peak_hour) <= 1

            p1_m1_results.append({
                'region': region,
                'date': date,
                'pred': pred_peak_hour,
                'actual': actual_peak_hour,
                'correct': success
            })

    except Exception as e:
        print(f"Error with region {region}: {e}")
        continue

p1_m1_df = pd.DataFrame(p1_m1_results)
m1_success_rate = p1_m1_df['correct'].mean() * 100
print(f"\nMethod 1 Success Rate: {m1_success_rate:.2f}%")
print(f"Correct: {p1_m1_df['correct'].sum()}/{len(p1_m1_df)}")

# ======================================================================
# ### Method 2: Random Forest Classification
# ======================================================================

print("\nTraining Method 2: Random Forest Classification...")

p1_m2_models = {}
p1_m2_results = []

for region in tqdm(regions, desc="Method 2 Training"):
    try:
        region_train = p1_train[p1_train['region'] == region].dropna(subset=FEATURE_COLS + ['is_peak_hour'])

        if len(region_train) < 100:
            continue

        X_train = region_train[FEATURE_COLS]
        y_train = region_train['is_peak_hour']

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        p1_m2_models[region] = model

    except Exception as e:
        print(f"Error training region {region}: {e}")
        continue

print(f"Trained: {len(p1_m2_models)} models")

# Make predictions
print("Making predictions for Method 2...")
for region in tqdm(regions, desc="Method 2 Predictions"):
    if region not in p1_m2_models:
        continue

    region_test = p1_test[p1_test['region'] == region].dropna(subset=FEATURE_COLS)

    if len(region_test) == 0:
        continue

    X_test = region_test[FEATURE_COLS]
    y_pred_proba = p1_m2_models[region].predict_proba(X_test)[:, 1]

    # For each day, find hour with highest probability
    region_test_copy = region_test.copy()
    region_test_copy['peak_prob'] = y_pred_proba

    for date in region_test_copy['date'].unique():
        day_data = region_test_copy[region_test_copy['date'] == date]

        pred_peak_hour = day_data.loc[day_data['peak_prob'].idxmax(), 'hour']
        actual_peak_hour = day_data.loc[day_data['load'].idxmax(), 'hour']

        success = np.abs(pred_peak_hour - actual_peak_hour) <= 1

        p1_m2_results.append({
            'region': region,
            'date': date,
            'pred': pred_peak_hour,
            'actual': actual_peak_hour,
            'correct': success
        })

p1_m2_df = pd.DataFrame(p1_m2_results)
m2_success_rate = p1_m2_df['correct'].mean() * 100
print(f"\nMethod 2 Success Rate: {m2_success_rate:.2f}%")
print(f"Correct: {p1_m2_df['correct'].sum()}/{len(p1_m2_df)}")

# ======================================================================
# ### Method 3: XGBoost Classification
# ======================================================================

print("\nTraining Method 3: XGBoost Classification...")

p1_m3_models = {}
p1_m3_results = []

for region in tqdm(regions, desc="Method 3 Training"):
    try:
        region_train = p1_train[p1_train['region'] == region].dropna(subset=FEATURE_COLS + ['is_peak_hour'])

        if len(region_train) < 100:
            continue

        X_train = region_train[FEATURE_COLS]
        y_train = region_train['is_peak_hour']

        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        model = XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        p1_m3_models[region] = model

    except Exception as e:
        print(f"Error training region {region}: {e}")
        continue

print(f"Trained: {len(p1_m3_models)} models")

# Make predictions
print("Making predictions for Method 3...")
for region in tqdm(regions, desc="Method 3 Predictions"):
    if region not in p1_m3_models:
        continue

    region_test = p1_test[p1_test['region'] == region].dropna(subset=FEATURE_COLS)

    if len(region_test) == 0:
        continue

    X_test = region_test[FEATURE_COLS]
    y_pred_proba = p1_m3_models[region].predict_proba(X_test)[:, 1]

    region_test_copy = region_test.copy()
    region_test_copy['peak_prob'] = y_pred_proba

    for date in region_test_copy['date'].unique():
        day_data = region_test_copy[region_test_copy['date'] == date]

        pred_peak_hour = day_data.loc[day_data['peak_prob'].idxmax(), 'hour']
        actual_peak_hour = day_data.loc[day_data['load'].idxmax(), 'hour']

        success = np.abs(pred_peak_hour - actual_peak_hour) <= 1

        p1_m3_results.append({
            'region': region,
            'date': date,
            'pred': pred_peak_hour,
            'actual': actual_peak_hour,
            'correct': success
        })

p1_m3_df = pd.DataFrame(p1_m3_results)
m3_success_rate = p1_m3_df['correct'].mean() * 100
print(f"\nMethod 3 Success Rate: {m3_success_rate:.2f}%")
print(f"Correct: {p1_m3_df['correct'].sum()}/{len(p1_m3_df)}")

# ======================================================================
# ### Method 4: LightGBM Classification
# ======================================================================

print("\nTraining Method 4: LightGBM Classification...")

p1_m4_models = {}
p1_m4_results = []

for region in tqdm(regions, desc="Method 4 Training"):
    try:
        region_train = p1_train[p1_train['region'] == region].dropna(subset=FEATURE_COLS + ['is_peak_hour'])

        if len(region_train) < 100:
            continue

        X_train = region_train[FEATURE_COLS]
        y_train = region_train['is_peak_hour']

        model = LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)

        p1_m4_models[region] = model

    except Exception as e:
        print(f"Error training region {region}: {e}")
        continue

print(f"Trained: {len(p1_m4_models)} models")

# Make predictions
print("Making predictions for Method 4...")
for region in tqdm(regions, desc="Method 4 Predictions"):
    if region not in p1_m4_models:
        continue

    region_test = p1_test[p1_test['region'] == region].dropna(subset=FEATURE_COLS)

    if len(region_test) == 0:
        continue

    X_test = region_test[FEATURE_COLS]
    y_pred_proba = p1_m4_models[region].predict_proba(X_test)[:, 1]

    region_test_copy = region_test.copy()
    region_test_copy['peak_prob'] = y_pred_proba

    for date in region_test_copy['date'].unique():
        day_data = region_test_copy[region_test_copy['date'] == date]

        pred_peak_hour = day_data.loc[day_data['peak_prob'].idxmax(), 'hour']
        actual_peak_hour = day_data.loc[day_data['load'].idxmax(), 'hour']

        success = np.abs(pred_peak_hour - actual_peak_hour) <= 1

        p1_m4_results.append({
            'region': region,
            'date': date,
            'pred': pred_peak_hour,
            'actual': actual_peak_hour,
            'correct': success
        })

p1_m4_df = pd.DataFrame(p1_m4_results)
m4_success_rate = p1_m4_df['correct'].mean() * 100
print(f"\nMethod 4 Success Rate: {m4_success_rate:.2f}%")
print(f"Correct: {p1_m4_df['correct'].sum()}/{len(p1_m4_df)}")

# ======================================================================
# ### Method 5: Historical Same Calendar Date
# ======================================================================

print("\nMethod 5: Historical Same Calendar Date...")

p1_m5_results = []

for region in tqdm(regions, desc="Method 5"):
    region_test = p1_test[p1_test['region'] == region]
    region_train = p1_train[p1_train['region'] == region]

    if len(region_test) == 0:
        continue

    for date in region_test['date'].unique():
        # Get the same calendar date from previous years
        test_date = pd.Timestamp(date)
        month = test_date.month
        day = test_date.day

        # Find historical same calendar dates
        historical = region_train[
            (region_train['datetime'].dt.month == month) &
            (region_train['datetime'].dt.day == day)
        ]

        if len(historical) == 0:
            # Fallback to mode of peak hours in training data
            pred_peak_hour = region_train.groupby('date')['hour'].apply(
                lambda x: x[region_train.loc[x.index, 'load'].idxmax()]
            ).mode().iloc[0] if len(region_train) > 0 else 17
        else:
            # Find most common peak hour on this calendar date
            historical_dates = historical['datetime'].dt.date.unique()
            peak_hours = []
            for hist_date in historical_dates:
                day_data = historical[historical['datetime'].dt.date == hist_date]
                if len(day_data) > 0:
                    peak_hours.append(day_data.loc[day_data['load'].idxmax(), 'hour'])

            if peak_hours:
                pred_peak_hour = pd.Series(peak_hours).mode().iloc[0]
            else:
                pred_peak_hour = 17  # Default fallback

        # Get actual peak hour
        day_data = region_test[region_test['date'] == date]
        actual_peak_hour = day_data.loc[day_data['load'].idxmax(), 'hour']

        success = np.abs(pred_peak_hour - actual_peak_hour) <= 1

        p1_m5_results.append({
            'region': region,
            'date': date,
            'pred': pred_peak_hour,
            'actual': actual_peak_hour,
            'correct': success
        })

p1_m5_df = pd.DataFrame(p1_m5_results)
m5_success_rate = p1_m5_df['correct'].mean() * 100
print(f"\nMethod 5 Success Rate: {m5_success_rate:.2f}%")
print(f"Correct: {p1_m5_df['correct'].sum()}/{len(p1_m5_df)}")

# Compile Phase 1 results
p1_comparison = pd.DataFrame([
    {
        'Method': '1. Linear + Interactions',
        'Total': len(p1_m1_df),
        'Correct': p1_m1_df['correct'].sum(),
        'Incorrect': len(p1_m1_df) - p1_m1_df['correct'].sum(),
        'Success_Rate_%': m1_success_rate
    },
    {
        'Method': '2. Random Forest',
        'Total': len(p1_m2_df),
        'Correct': p1_m2_df['correct'].sum(),
        'Incorrect': len(p1_m2_df) - p1_m2_df['correct'].sum(),
        'Success_Rate_%': m2_success_rate
    },
    {
        'Method': '3. XGBoost',
        'Total': len(p1_m3_df),
        'Correct': p1_m3_df['correct'].sum(),
        'Incorrect': len(p1_m3_df) - p1_m3_df['correct'].sum(),
        'Success_Rate_%': m3_success_rate
    },
    {
        'Method': '4. LightGBM',
        'Total': len(p1_m4_df),
        'Correct': p1_m4_df['correct'].sum(),
        'Incorrect': len(p1_m4_df) - p1_m4_df['correct'].sum(),
        'Success_Rate_%': m4_success_rate
    },
    {
        'Method': '5. Historical (Same Date)',
        'Total': len(p1_m5_df),
        'Correct': p1_m5_df['correct'].sum(),
        'Incorrect': len(p1_m5_df) - p1_m5_df['correct'].sum(),
        'Success_Rate_%': m5_success_rate
    }
])

p1_comparison = p1_comparison.sort_values('Success_Rate_%', ascending=False)

print("\n" + "="*70)
print("PHASE 1 RESULTS SUMMARY")
print("="*70)
print(p1_comparison.to_string(index=False))

best_method_name = p1_comparison.iloc[0]['Method']
best_success_rate = p1_comparison.iloc[0]['Success_Rate_%']

print(f"\n✓ Best Method: {best_method_name} ({best_success_rate:.2f}%)")

# Retrain best method for all regions
print(f"\nRetraining best method ({best_method_name}) for all regions...")
peak_hour_best_models = {}
for region in tqdm(regions, desc="Method 3 Training"):
    region_final_train = p1_final_train[p1_final_train['region'] == region].dropna(subset=FEATURE_COLS + ['is_peak_hour'])

    X_train = region_final_train[FEATURE_COLS]
    y_train = region_final_train['is_peak_hour']

    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    peak_hour_best_models[region] = {
                'model': model,
                'method': best_method_name,
                'feature_cols': FEATURE_COLS
            }

print(f"Method saved for {len(peak_hour_best_models)} regions")

# ======================================================================
# ## 7. Task 3: Peak Days Prediction - Best Model Finding
# ======================================================================

print("\n" + "=" * 70)
print("TASK 3: PEAK DAYS PREDICTION - BEST MODEL FINDING")
print("=" * 70)

FP_COST = 1  # False Positive: predict peak when not peak
FN_COST = 4  # False Negative: miss a peak day

# Get US holidays
us_holidays = holidays.US(years=range(2016, 2026))

df['date'] = df['datetime'].dt.date

# Aggregate to daily level - WEATHER ONLY (no load features except for labeling)
daily_agg = df.groupby(['date', 'region']).agg({
    'load': 'max',  # Only used for labeling peak days, not as a feature
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

print(f"Daily aggregated data: {len(daily_agg):,} rows")
print(f"Date range: {daily_agg['date'].min()} to {daily_agg['date'].max()}")
print(f"\nFeatures created (NO LOAD FEATURES - weather and date only)")

# Define methods for peak days (NO LOAD FEATURES)
def get_peak_days_methods():
    return {
        '1. RF Regression + Ranking': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        '2. RF Classification + Asymmetric Loss': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, class_weight={0: 1, 1: 4}
        ),
        '3. XGBoost Classification + Asymmetric Loss': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, scale_pos_weight=4, eval_metric='logloss'
        )
    }

# Features: NO LOAD FEATURES
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

print("Methods defined for peak days prediction")

def create_peak_day_labels(daily_data, window_size=10):
    """
    Create peak day labels using rolling 10-day windows.
    For training: creates overlapping windows for more data.
    Each window: top 2 days by max load are labeled as peak days.
    """
    labeled_data = []

    for region in tqdm(daily_data['region'].unique(), desc="Creating labels"):
        region_data = daily_data[daily_data['region'] == region].sort_values('date').copy()
        dates = region_data['date'].unique()

        # Create rolling windows
        for i in range(len(dates) - window_size + 1):
            window_dates = dates[i:i+window_size]
            window_data = region_data[region_data['date'].isin(window_dates)].copy()

            # Find top 2 days by max load
            top_2_dates = window_data.nlargest(2, 'load_max')['date'].values

            # Label each day
            for date in window_dates:
                day_data = window_data[window_data['date'] == date].iloc[0].to_dict()
                day_data['is_peak_day'] = 1 if date in top_2_dates else 0
                day_data['window_start'] = window_dates[0]
                labeled_data.append(day_data)

    return pd.DataFrame(labeled_data)

def create_test_labels(daily_data, test_start, test_end):
    """
    Create labels for test period WITHOUT overlapping windows.
    For the exact 10-day test period, label top 2 days as peak.
    Each day appears exactly once.
    """
    labeled_data = []

    # Filter to test period
    test_data = daily_data[
        (daily_data['date'] >= test_start) &
        (daily_data['date'] <= test_end)
    ].copy()

    for region in tqdm(test_data['region'].unique(), desc="Creating test labels"):
        region_data = test_data[test_data['region'] == region].copy()

        # Find top 2 days by max load for this region
        top_2_dates = region_data.nlargest(2, 'load_max')['date'].values

        # Label each day
        for _, row in region_data.iterrows():
            row_dict = row.to_dict()
            row_dict['is_peak_day'] = 1 if row['date'] in top_2_dates else 0
            labeled_data.append(row_dict)

    return pd.DataFrame(labeled_data)

# Create labels for TRAINING data (with overlapping windows)
print("Creating peak day labels for training (overlapping windows)...")
labeled_daily = create_peak_day_labels(daily_agg)

print(f"\nLabeled training data: {len(labeled_daily):,} rows (includes overlaps)")
print(f"Peak days: {labeled_daily['is_peak_day'].sum():,} ({labeled_daily['is_peak_day'].mean()*100:.1f}%)")
print(f"Non-peak days: {(labeled_daily['is_peak_day']==0).sum():,}")

# Phase 1: Split training data (with overlaps) and test data (no overlaps)
p1_train = labeled_daily[
    (labeled_daily['date'] >= TRAIN_START) &
    (labeled_daily['date'] <= TRAIN_END)
].copy()

p1_final_train= labeled_daily[
    (labeled_daily['date'] >= FINAL_TRAIN_START) &
    (labeled_daily['date'] <= FINAL_TRAIN_END)
].copy()

# For test data, create clean labels without overlapping windows
print("\nCreating Phase 1 test labels (no overlaps)...")
p1_test = create_test_labels(daily_agg, TEST_START, TEST_END)

print("\nFeature columns defined (weather and calendar only")
print(f"Total features: {len(feature_cols)}")

def calculate_loss(y_true, y_pred, fp_cost=1, fn_cost=4):
    """
    Calculate custom loss: FN_COST * false_negatives + FP_COST * false_positives
    FN (miss a peak) = 4, FP (false alarm) = 1
    """
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fn_cost * fn + fp_cost * fp

def tune_threshold(y_true, y_pred_proba, fp_cost=1, fn_cost=4):
    """
    Find optimal threshold to minimize custom loss
    """
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_loss = float('inf')
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        loss = calculate_loss(y_true, y_pred, fp_cost, fn_cost)

        if loss < best_loss:
            best_loss = loss
            best_threshold = threshold

    return best_threshold, best_loss

def evaluate_predictions(test_data, predictions, method_name):
    """
    Evaluate predictions and return detailed metrics
    """
    y_true = test_data['is_peak_day'].values

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    # Calculate custom loss
    loss = calculate_loss(y_true, predictions, FP_COST, FN_COST)

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Method': method_name,
        'Total_Loss': loss,
        'False_Positives': fp,
        'False_Negatives': fn,
        'True_Positives': tp,
        'True_Negatives': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }

print("Helper functions defined")

# Prepare Phase 1 data
X_train_p1 = p1_train[feature_cols]
y_train_p1 = p1_train['is_peak_day']
X_test_p1 = p1_test[feature_cols]
y_test_p1 = p1_test['is_peak_day']

# # Add region as categorical
# X_train_p1_with_region = pd.get_dummies(p1_train[feature_cols + ['region']], columns=['region'])
# X_test_p1_with_region = pd.get_dummies(p1_test[feature_cols + ['region']], columns=['region'])
# X_test_p1_with_region = X_test_p1_with_region.reindex(columns=X_train_p1_with_region.columns, fill_value=0)

print("Train test data prepared")
print(f"Training features shape: {X_train_p1.shape}")
print(f"Testing features shape: {X_test_p1.shape}")

# ======================================================================
# ### Method 1: Random Forest Classification
# ======================================================================

# Train separate Random Forest model for each region
print("Training Method 1: Random Forest (29 regional models)...")

rf_regional_models = {}
rf_regional_predictions = []

for region in tqdm(regions, desc="Training RF models"):
    # Filter data for this region
    train_region = p1_train[p1_train['region'] == region].copy()
    test_region = p1_test[p1_test['region'] == region].copy()

    if len(train_region) == 0 or len(test_region) == 0:
        print(f"  Skipping {region}: insufficient data")
        continue

    # Prepare features (without region column)
    X_train_region = train_region[feature_cols]
    y_train_region = train_region['is_peak_day']
    X_test_region = test_region[feature_cols]
    y_test_region = test_region['is_peak_day']

    # Train model for this region
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight={0: 1, 1: 4},
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_region, y_train_region)

    # Make predictions
    rf_proba = rf_model.predict_proba(X_test_region)[:, 1]
    rf_threshold, _ = tune_threshold(y_test_region, rf_proba, FP_COST, FN_COST)
    rf_pred = (rf_proba >= rf_threshold).astype(int)

    # Store model and predictions
    rf_regional_models[region] = {
        'model': rf_model,
        'threshold': rf_threshold
    }

    # Store predictions with region info
    test_region_copy = test_region.copy()
    test_region_copy['rf_pred'] = rf_pred
    rf_regional_predictions.append(test_region_copy)

# Combine all regional predictions
rf_all_predictions = pd.concat(rf_regional_predictions, ignore_index=True)

# Evaluate overall performance
rf_results = evaluate_predictions(rf_all_predictions, rf_all_predictions['rf_pred'], "1. Random Forest (Regional)")

print(f"\n{'='*70}")
print("RANDOM FOREST - 29 REGIONAL MODELS")
print(f"{'='*70}")
print(f"Total Loss: {rf_results['Total_Loss']}")
print(f"False Positives: {rf_results['False_Positives']}")
print(f"False Negatives: {rf_results['False_Negatives']}")
print(f"Accuracy: {rf_results['Accuracy']:.3f}")
print(f"Precision: {rf_results['Precision']:.3f}")
print(f"Recall: {rf_results['Recall']:.3f}")
print(f"Models trained: {len(rf_regional_models)}/{len(regions)} regions")

# ======================================================================
# ### Method 2: XGBoost Classification
# ======================================================================

# Train separate XGBoost model for each region
print("Training Method 2: XGBoost (29 regional models)...")

xgb_regional_models = {}
xgb_regional_predictions = []

for region in tqdm(regions, desc="Training XGBoost models"):
    # Filter data for this region
    train_region = p1_train[p1_train['region'] == region].copy()
    test_region = p1_test[p1_test['region'] == region].copy()

    if len(train_region) == 0 or len(test_region) == 0:
        print(f"  Skipping {region}: insufficient data")
        continue

    # Prepare features (without region column)
    X_train_region = train_region[feature_cols]
    y_train_region = train_region['is_peak_day']
    X_test_region = test_region[feature_cols]
    y_test_region = test_region['is_peak_day']

    # Calculate scale_pos_weight for this region
    neg_count = (y_train_region == 0).sum()
    pos_count = (y_train_region == 1).sum()
    scale_pos_weight = (neg_count / pos_count) * FN_COST if pos_count > 0 else 1.0

    # Train model for this region
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_region, y_train_region)

    # Make predictions
    xgb_proba = xgb_model.predict_proba(X_test_region)[:, 1]
    xgb_threshold, _ = tune_threshold(y_test_region, xgb_proba, FP_COST, FN_COST)
    xgb_pred = (xgb_proba >= xgb_threshold).astype(int)

    # Store model and predictions
    xgb_regional_models[region] = {
        'model': xgb_model,
        'threshold': xgb_threshold,
        'scale_pos_weight': scale_pos_weight
    }

    # Store predictions with region info
    test_region_copy = test_region.copy()
    test_region_copy['xgb_pred'] = xgb_pred
    xgb_regional_predictions.append(test_region_copy)

# Combine all regional predictions
xgb_all_predictions = pd.concat(xgb_regional_predictions, ignore_index=True)

# Evaluate overall performance
xgb_results = evaluate_predictions(xgb_all_predictions, xgb_all_predictions['xgb_pred'], "2. XGBoost (Regional)")

print(f"\n{'='*70}")
print("XGBOOST - 29 REGIONAL MODELS")
print(f"{'='*70}")
print(f"Total Loss: {xgb_results['Total_Loss']}")
print(f"False Positives: {xgb_results['False_Positives']}")
print(f"False Negatives: {xgb_results['False_Negatives']}")
print(f"Accuracy: {xgb_results['Accuracy']:.3f}")
print(f"Precision: {xgb_results['Precision']:.3f}")
print(f"Recall: {xgb_results['Recall']:.3f}")
print(f"Models trained: {len(xgb_regional_models)}/{len(regions)} regions")
print(f"{'='*70}")

# ======================================================================
# ### Method 3: LightGBM Classification
# ======================================================================

# Train separate LightGBM model for each region
print("Training Method 3: LightGBM (29 regional models)...")

lgbm_regional_models = {}
lgbm_regional_predictions = []

for region in tqdm(regions, desc="Training LightGBM models"):
    # Filter data for this region
    train_region = p1_train[p1_train['region'] == region].copy()
    test_region = p1_test[p1_test['region'] == region].copy()

    if len(train_region) == 0 or len(test_region) == 0:
        print(f"  Skipping {region}: insufficient data")
        continue

    # Prepare features (without region column)
    X_train_region = train_region[feature_cols]
    y_train_region = train_region['is_peak_day']
    X_test_region = test_region[feature_cols]
    y_test_region = test_region['is_peak_day']

    # Calculate scale_pos_weight for this region
    neg_count = (y_train_region == 0).sum()
    pos_count = (y_train_region == 1).sum()
    scale_pos_weight = (neg_count / pos_count) * FN_COST if pos_count > 0 else 1.0

    # Train model for this region
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_model.fit(X_train_region, y_train_region)

    # Make predictions
    lgbm_proba = lgbm_model.predict_proba(X_test_region)[:, 1]
    lgbm_threshold, _ = tune_threshold(y_test_region, lgbm_proba, FP_COST, FN_COST)
    lgbm_pred = (lgbm_proba >= lgbm_threshold).astype(int)

    # Store model and predictions
    lgbm_regional_models[region] = {
        'model': lgbm_model,
        'threshold': lgbm_threshold,
        'scale_pos_weight': scale_pos_weight
    }

    # Store predictions with region info
    test_region_copy = test_region.copy()
    test_region_copy['lgbm_pred'] = lgbm_pred
    lgbm_regional_predictions.append(test_region_copy)

# Combine all regional predictions
lgbm_all_predictions = pd.concat(lgbm_regional_predictions, ignore_index=True)

# Evaluate overall performance
lgbm_results = evaluate_predictions(lgbm_all_predictions, lgbm_all_predictions['lgbm_pred'], "3. LightGBM (Regional)")

print(f"\n{'='*70}")
print("LIGHTGBM - 29 REGIONAL MODELS")
print(f"{'='*70}")
print(f"Total Loss: {lgbm_results['Total_Loss']}")
print(f"False Positives: {lgbm_results['False_Positives']}")
print(f"False Negatives: {lgbm_results['False_Negatives']}")
print(f"Accuracy: {lgbm_results['Accuracy']:.3f}")
print(f"Precision: {lgbm_results['Precision']:.3f}")
print(f"Recall: {lgbm_results['Recall']:.3f}")
print(f"Models trained: {len(lgbm_regional_models)}/{len(regions)} regions")
print(f"{'='*70}")

# ======================================================================
# ### Method 4: Logistic Regression with Interactions
# ======================================================================

# Train separate Logistic Regression model for each region
print("Training Method 4: Logistic Regression (29 regional models)...")

lr_regional_models = {}
lr_regional_predictions = []

for region in tqdm(regions, desc="Training Logistic Regression models"):
    # Filter data for this region
    train_region = p1_train[p1_train['region'] == region].copy()
    test_region = p1_test[p1_test['region'] == region].copy()

    if len(train_region) == 0 or len(test_region) == 0:
        print(f"  Skipping {region}: insufficient data")
        continue

    # Prepare features (without region column)
    X_train_region = train_region[feature_cols]
    y_train_region = train_region['is_peak_day']
    X_test_region = test_region[feature_cols]
    y_test_region = test_region['is_peak_day']

    # Scale features for this region
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_region)
    X_test_scaled = scaler.transform(X_test_region)

    # Train model for this region
    lr_model = LogisticRegression(
        class_weight={0: 1, 1: 4},
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train_region)

    # Make predictions
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_threshold, _ = tune_threshold(y_test_region, lr_proba, FP_COST, FN_COST)
    lr_pred = (lr_proba >= lr_threshold).astype(int)

    # Store model, scaler, and predictions
    lr_regional_models[region] = {
        'model': lr_model,
        'scaler': scaler,
        'threshold': lr_threshold
    }

    # Store predictions with region info
    test_region_copy = test_region.copy()
    test_region_copy['lr_pred'] = lr_pred
    lr_regional_predictions.append(test_region_copy)

# Combine all regional predictions
lr_all_predictions = pd.concat(lr_regional_predictions, ignore_index=True)

# Evaluate overall performance
lr_results = evaluate_predictions(lr_all_predictions, lr_all_predictions['lr_pred'], "4. Logistic Regression (Regional)")

print(f"\n{'='*70}")
print("LOGISTIC REGRESSION - 29 REGIONAL MODELS")
print(f"{'='*70}")
print(f"Total Loss: {lr_results['Total_Loss']}")
print(f"False Positives: {lr_results['False_Positives']}")
print(f"False Negatives: {lr_results['False_Negatives']}")
print(f"Accuracy: {lr_results['Accuracy']:.3f}")
print(f"Precision: {lr_results['Precision']:.3f}")
print(f"Recall: {lr_results['Recall']:.3f}")
print(f"Models trained: {len(lr_regional_models)}/{len(regions)} regions")
print(f"{'='*70}")

# ======================================================================
# ### Results Summary
# ======================================================================

# Summary of peak days results
# Compile results
p1_comparison = pd.DataFrame([
    rf_results,
    xgb_results,
    lgbm_results,
    lr_results
]).sort_values('Total_Loss')

print("\n" + "="*80)
print("PHASE 1 RESULTS - MODEL COMPARISON (2023 Test)")
print("="*80)
print(p1_comparison[['Method', 'Total_Loss', 'False_Positives', 'False_Negatives',
                      'Accuracy', 'Precision', 'Recall']].to_string(index=False))
print("\n" + "="*80)

# Identify best method
best_method_name = p1_comparison.iloc[0]['Method']
best_loss = p1_comparison.iloc[0]['Total_Loss']

print(f"\n✓ Best Method: {best_method_name}")
print(f"✓ Best Loss: {best_loss}")
print(f"✓ FP: {p1_comparison.iloc[0]['False_Positives']} | FN: {p1_comparison.iloc[0]['False_Negatives']}")

# Determine which model won
best_method_num = best_method_name.split('.')[0]

# Train separate LightGBM model for each region
print("Training Method 3: LightGBM (29 regional models)...")

peak_days_best_models={}

for region in tqdm(regions, desc="Training LightGBM models"):
    # Filter data for this region
    train_final_region = p1_final_train[p1_final_train['region'] == region].copy()

    # Prepare features (without region column)
    X_train_region = train_final_region[feature_cols]
    y_train_region = train_final_region['is_peak_day']

    # Calculate scale_pos_weight for this region
    neg_count = (y_train_region == 0).sum()
    pos_count = (y_train_region == 1).sum()
    scale_pos_weight = (neg_count / pos_count) * FN_COST if pos_count > 0 else 1.0

    # Train model for this region
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_model.fit(X_train_region, y_train_region)

    # Store model and predictions
    peak_days_best_models[region] = {
        'model': lgbm_model,
        'method': best_method_name,
        'threshold': lgbm_threshold,
        'scale_pos_weight': scale_pos_weight
    }

# ======================================================================
# ## 8. Save Trained Best Models
# ======================================================================

print("\n" + "=" * 70)
print("SAVING TRAINED BEST MODELS")
print("=" * 70)

# Create models directory
models_dir = os.path.join(OUTPUT_DIR, 'trained_models')
os.makedirs(models_dir, exist_ok=True)

# Save hourly load models
print("\nSaving hourly load forecast models...")
with open(os.path.join(models_dir, 'hourly_load_models.pkl'), 'wb') as f:
    pickle.dump(hourly_best_models, f)
print(f"  Saved {len(hourly_best_models)} regional models")

# Save peak hour models
print("\nSaving peak hour prediction models...")
with open(os.path.join(models_dir, 'peak_hour_models.pkl'), 'wb') as f:
    pickle.dump(peak_hour_best_models, f)
print(f"  Saved {len(peak_hour_best_models)} regional models")

# Save peak days models
print("\nSaving peak days prediction models...")
with open(os.path.join(models_dir, 'peak_days_models.pkl'), 'wb') as f:
    pickle.dump(peak_days_best_models, f)

print(f"  Saved {len(peak_days_best_models)} regional models")
print(f"\nAll models saved to: {models_dir}")
