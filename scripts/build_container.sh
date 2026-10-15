#!/bin/bash
set -e

APP_DIR="/home/ec2-user/ml-deployment"

echo "Building ML Prediction API Docker image..."

cd "$APP_DIR"
docker build -t ml-prediction-api:1.0 .

echo "Docker image built successfully."
