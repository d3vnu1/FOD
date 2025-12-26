"""
Main entry point for FleeKey Object Detection service.
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path

from .config import load_config
from .capture import VideoCapture
from .detector import YOLOv8Detector
from .streamer import MJPEGStreamer
from .utils import draw_detections, get_coco_class_names


# Global shutdown flag
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_flag
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_flag = True


def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration.

    Args:
        level: Logging level
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Module-level logger (initialized after setup_logging is called)
logger = None


def main():
    """Main application loop."""
    global logger, shutdown_flag

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FleeKey Object Detection Service')
    parser.add_argument(
        '-c', '--config',
        default='/etc/hailo-detector/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # Setup logging
    log_level = "DEBUG" if args.debug else config.logging.level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("FleeKey Object Detection v1.1.0")
    logger.info("=" * 70)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize components
    video_capture = None
    detector = None
    streamer = None

    try:
        # Initialize video capture
        logger.info("Initializing video capture...")
        video_capture = VideoCapture(config.video)

        # Initialize detector
        logger.info("Initializing YOLOv8 detector...")
        detector = YOLOv8Detector(config.inference)

        if not detector.is_ready:
            logger.error("Failed to initialize detector, but continuing...")
            logger.error("Frames will be streamed without detections")

        # Initialize MJPEG streamer
        logger.info("Initializing MJPEG streamer...")
        streamer = MJPEGStreamer(config.stream)
        streamer.start()

        # Give server time to start
        time.sleep(1)

        logger.info(f"Stream available at http://<your-ip>:{config.stream.port}/")
        logger.info("Starting main processing loop...")

        # Open camera
        if not video_capture.open():
            logger.error("Failed to open camera initially, will retry...")

        # Main processing loop
        run_main_loop(video_capture, detector, streamer)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        logger.info("Cleaning up...")

        if video_capture is not None:
            video_capture.release()

        if detector is not None:
            detector.cleanup()

        if streamer is not None:
            streamer.stop()

        logger.info("Shutdown complete")

    return 0


def run_main_loop(video_capture: VideoCapture, detector: YOLOv8Detector,
                  streamer: MJPEGStreamer):
    """
    Main processing loop.

    Args:
        video_capture: Video capture instance
        detector: Detector instance
        streamer: Streamer instance
    """
    global shutdown_flag

    class_names = get_coco_class_names()

    # FPS calculation
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    last_stats_log_time = time.time()

    while not shutdown_flag:
        # Check if camera is opened
        if not video_capture.is_opened:
            logger.warning("Camera not available, attempting to reconnect...")
            if not video_capture.reconnect():
                # Use last frame if available, or create a placeholder
                last_frame = video_capture.get_last_frame()
                if last_frame is not None:
                    streamer.update_frame(last_frame)
                continue

        # Capture frame
        frame = video_capture.read()

        if frame is None:
            logger.warning("Failed to read frame")
            continue

        # Detect objects
        inference_start = time.time()
        boxes, scores, class_ids = detector.detect(frame)
        inference_time = time.time() - inference_start

        # Draw detections
        annotated_frame = draw_detections(frame, boxes, scores, class_ids, class_names)

        # Update streamer
        streamer.update_frame(annotated_frame)

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # Update statistics
        streamer.update_stats(fps, inference_time, len(boxes))

        # Log statistics every 30 seconds
        if time.time() - last_stats_log_time >= 30.0:
            logger.info(
                f"Stats: FPS={fps:.1f}, Inference={inference_time*1000:.1f}ms, "
                f"Detections={len(boxes)}"
            )
            last_stats_log_time = time.time()

        # Small delay to prevent busy loop (allow other threads to run)
        time.sleep(0.001)


if __name__ == '__main__':
    sys.exit(main())
