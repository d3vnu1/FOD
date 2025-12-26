#!/usr/bin/env python3
"""
Test script to debug Hailo inference with a single frame.
"""

import cv2
import numpy as np
import sys
import logging

# Add the source directory to path
sys.path.insert(0, '/opt/hailo-detector/venv/lib/python3.11/site-packages')

from hailo_detector.inference import HailoInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize inference
    model_path = "/opt/hailo-detector/models/yolov8n.hef"
    inference = HailoInference(model_path)

    if not inference.initialize():
        logger.error("Failed to initialize inference")
        return 1

    # Capture a frame from camera
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Failed to capture frame")
        return 1

    logger.info(f"Captured frame: {frame.shape}, dtype: {frame.dtype}")

    # Save the frame for debugging
    cv2.imwrite("/tmp/test_frame.jpg", frame)
    logger.info("Saved test frame to /tmp/test_frame.jpg")

    # Letterbox to 640x640
    from hailo_detector.utils import letterbox_image
    letterboxed, scale, pad = letterbox_image(frame, 640)
    logger.info(f"Letterboxed: {letterboxed.shape}, scale: {scale}, pad: {pad}")
    cv2.imwrite("/tmp/test_letterboxed.jpg", letterboxed)

    # Preprocess
    preprocessed = inference.preprocess(letterboxed)
    logger.info(f"Preprocessed: {preprocessed.shape}, dtype: {preprocessed.dtype}")
    logger.info(f"Preprocessed range: [{preprocessed.min()}, {preprocessed.max()}]")
    logger.info(f"Preprocessed mean: {preprocessed.mean()}, std: {preprocessed.std()}")

    # Try inference with original preprocessing
    logger.info("\n=== Test 1: NHWC format, FLOAT32, [0,1] ===")
    results = inference.infer(preprocessed)
    if results:
        for key, value in results.items():
            if isinstance(value, list):
                value = np.array(value)
            logger.info(f"Output '{key}': shape={value.shape}, dtype={value.dtype}")
            # Count non-zero detections
            if len(value.shape) == 4:
                num_detections = value.shape[2]
                logger.info(f"  Number of detections: {num_detections}")

    # Try NCHW format (transpose)
    logger.info("\n=== Test 2: NCHW format (transposed) ===")
    preprocessed_nchw = preprocessed.transpose(0, 3, 1, 2)  # (1,640,640,3) -> (1,3,640,640)
    logger.info(f"NCHW shape: {preprocessed_nchw.shape}")
    try:
        results = inference.infer(preprocessed_nchw)
        if results:
            for key, value in results.items():
                if isinstance(value, list):
                    value = np.array(value)
                logger.info(f"Output '{key}': shape={value.shape}")
    except Exception as e:
        logger.error(f"NCHW format failed: {e}")

    # Try with different normalization
    logger.info("\n=== Test 3: Different normalization ranges ===")

    # Test with [0, 255] uint8
    image_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.uint8)
    image_batch = np.expand_dims(image_rgb, axis=0)
    logger.info(f"UINT8 format: {image_batch.shape}, {image_batch.dtype}, range=[{image_batch.min()}, {image_batch.max()}]")

    # Try [-1, 1] normalization
    image_normalized_pm1 = (image_rgb.astype(np.float32) / 127.5) - 1.0
    image_normalized_pm1 = np.expand_dims(image_normalized_pm1, axis=0)
    logger.info(f"[-1, 1] format: {image_normalized_pm1.shape}, {image_normalized_pm1.dtype}, range=[{image_normalized_pm1.min():.3f}, {image_normalized_pm1.max():.3f}]")
    results = inference.infer(image_normalized_pm1)
    if results:
        for key, value in results.items():
            if isinstance(value, list):
                value = np.array(value)
            logger.info(f"Output '{key}': shape={value.shape}")
            if len(value.shape) == 4:
                logger.info(f"  Number of detections: {value.shape[2]}")

    logger.info("\nTest complete!")
    inference.cleanup()

    return 0

if __name__ == "__main__":
    sys.exit(main())
