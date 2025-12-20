#!/bin/sh
# Download YOLOv8n HEF model for Hailo-8L from Hailo Model Zoo

set -e

MODEL_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8l/yolov8n.hef"
MODEL_DIR="/opt/hailo-detector/models"
MODEL_FILE="$MODEL_DIR/yolov8n.hef"

echo "Hailo Model Download Script"
echo "============================"
echo ""

# Create model directory if it doesn't exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating model directory: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
fi

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at: $MODEL_FILE"
    echo "File size: $(du -h "$MODEL_FILE" | cut -f1)"
    echo ""
    read -p "Do you want to re-download? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Keeping existing model."
        exit 0
    fi
    echo "Removing existing model..."
    rm -f "$MODEL_FILE"
fi

# Download model
echo "Downloading YOLOv8n HEF model for Hailo-8L..."
echo "URL: $MODEL_URL"
echo "Destination: $MODEL_FILE"
echo ""

if command -v curl >/dev/null 2>&1; then
    curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_FILE" "$MODEL_URL"
else
    echo "ERROR: Neither curl nor wget is available"
    echo "Please install curl or wget and try again"
    exit 1
fi

# Verify download
if [ -f "$MODEL_FILE" ]; then
    echo ""
    echo "Download complete!"
    echo "Model saved to: $MODEL_FILE"
    echo "File size: $(du -h "$MODEL_FILE" | cut -f1)"
    echo ""

    # Verify it's a HEF file (should start with specific bytes)
    file_type=$(file "$MODEL_FILE" 2>/dev/null || echo "unknown")
    echo "File type: $file_type"

    exit 0
else
    echo ""
    echo "ERROR: Download failed"
    echo "The model file was not created at: $MODEL_FILE"
    exit 1
fi
