#!/bin/bash
set -e

echo "Stopping existing ML API container..."

if docker ps -a --format '{{.Names}}' | grep -qx 'ml-api-container'; then
    docker rm -f ml-api-container
else
    echo "No existing ML API container to stop."
fi

echo "ApplicationStop completed."
