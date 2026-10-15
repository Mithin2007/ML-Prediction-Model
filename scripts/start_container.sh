#!/bin/bash
set -e

echo "Starting ML Prediction API container..."

docker run -d \
  --name ml-api-container \
  --restart unless-stopped \
  -p 127.0.0.1:8001:8000 \
  ml-prediction-api:1.0

echo "ML Prediction API container started."
