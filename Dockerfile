# Use Python base image instead of jupyter/r-notebook since your project uses Python
# For linux/amd64 platform support
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    make \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p data/raw data/processed output/predictions output/trained_models output/figures

# Set default command to bash for interactive use
CMD ["/bin/bash"]
