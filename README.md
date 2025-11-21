# PJM Electrical Load Forecasting

A comprehensive machine learning pipeline for predicting electrical load across 29 regions in the PJM (Pennsylvania-New Jersey-Maryland) interconnection grid system. This project includes hourly load forecasting, peak hour identification, and peak day prediction using multiple gradient boosting algorithms.

## Project Overview

This system predicts:
1. **Hourly Load** - Electrical demand for each hour (00-23) across 29 zones
2. **Peak Hour** - The hour with maximum load for each zone
3. **Peak Day** - Whether a day will be a peak demand day (binary classification)

The models are trained on historical load data (2016-2025) combined with weather features from the Open-Meteo API.

## Quick Start with Docker

### Pull and Run
```bash
# Pull the Docker image
docker pull dhruban/load-forecasting

# Generate predictions for tomorrow
docker run -it --rm dhruban/load-forecasting make predictions

# Save predictions to file
docker run -it --rm dhruban/load-forecasting make predictions >> predictions.csv
```

### Available Commands
```bash
# Get interactive bash terminal
docker run -it --rm dhruban/load-forecasting

# Inside the container, you can run:
make              # Run the full analysis pipeline
make predictions  # Generate tomorrow's predictions
make clean        # Remove all outputs except raw data and source code
make rawdata      # Re-download raw data from sources
```

## Prediction Output Format

The `make predictions` command outputs a single CSV line:
```
"YYYY-MM-DD", L1_00, L1_01, ..., L1_23, L2_00, ..., L29_23, PH_1, ..., PH_29, PD_1, ..., PD_29
```

Where:
- `YYYY-MM-DD` - Prediction date
- `L{zone}_{hour}` - Predicted load in megawatts (rounded to nearest integer)
  - 29 zones × 24 hours = 696 values
- `PH_{zone}` - Predicted peak hour (00-23) for each zone
  - 29 values
- `PD_{zone}` - Peak day prediction (0=no, 1=yes) for each zone
  - 29 values

**Total: 755 comma-separated values**

```
## Data Sources
- **Load Data**: PJM hourly metered load (2016-2025) via OSF
- **Weather Data**: Open-Meteo API forecasts

### Prerequisites
- Python 3.11+
- Docker (optional)

### Setup
```bash
# Clone the repository
git clone https://github.com/Dhruban/STATS604_Project4.git
cd STATS604_Project4

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python src/analysis_script.py

# Generate predictions
python src/prediction_script.py
```

### Requirements
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-dateutil>=2.8.2
pytz>=2023.3
holidays>=0.35
requests>=2.31.0
tqdm>=4.66.0
```

## Building the Docker Image

```bash
# Build for linux/amd64 platform
docker build --platform linux/amd64 -t dhruban/load-forecasting .

# Push to DockerHub
docker push dhruban/load-forecasting
```
## Daily Workflow

For operational use, run daily before noon:
```bash
# Generate today's predictions
docker run -it --rm dhruban/load-forecasting make predictions >> predictions.csv
```

## Author

**Dhruban Nandi**
- DockerHub: [@dhruban](https://hub.docker.com/u/dhruban)
- GitHub: [@Dhruban](https://github.com/Dhruban)

