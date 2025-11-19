.PHONY: all clean predictions rawdata help

# Default target: run analysis
all:
	python src/analysis_script.py

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Run analysis script"
	@echo "  make clean    - Delete everything except src/ and data/raw/"
	@echo "  make predictions - Make current predictions and output to screen"
	@echo "  make rawdata  - Delete and re-download raw data"

# Clean: delete everything except src/ and data/raw/
clean:
	@echo "Cleaning all outputs except src/ and data/raw/..."
	rm -rf data/processed
	rm -rf output
	rm -rf figures
	@echo "Clean complete. Only src/ and data/raw/ preserved."

# Download raw data
rawdata:
	@echo "Downloading raw data..."
	rm -rf data/raw/*
	@echo "Downloading weather data..."
	python src/weather_download.py
	@echo "Downloading load data from OSF..."
	curl -L "https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip=" -o data/raw/load_data.zip
	@echo "Extracting load data..."
	cd data/raw && unzip -q load_data.zip && rm load_data.zip
	@echo "Raw data download complete."

# Make predictions - output only the prediction line, no other output
predictions:
	@python src/prediction_script.py
