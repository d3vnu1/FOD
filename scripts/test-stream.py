#!/usr/bin/env python3
"""
Stream test utility to run the full pipeline for testing without systemd.
"""

import sys
import time
import signal
import os

# Add src to path for testing before installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hailo_detector.config import load_config, Config
from hailo_detector.capture import VideoCapture
from hailo_detector.detector import YOLOv8Detector
from hailo_detector.streamer import MJPEGStreamer
from hailo_detector.utils import draw_detections, get_coco_class_names
import logging


# Shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    print("\nShutdown requested...")
    shutdown_requested = True


def get_local_ip():
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    """Main test function."""
    global shutdown_requested

    print("=" * 70)
    print("Hailo Detector - Stream Test Utility")
    print("=" * 70)
    print()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Load configuration (use defaults if config doesn't exist)
    try:
        config = load_config()
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"Note: Using default configuration ({e})")
        config = Config()

    print()

    # Initialize components
    video_capture = None
    detector = None
    streamer = None

    try:
        # Initialize video capture
        print("Initializing video capture...")
        video_capture = VideoCapture(config.video)

        if not video_capture.open():
            print("✗ Failed to open camera")
            print(f"  Make sure {config.video.device} is available")
            return 1

        print(f"✓ Camera opened: {config.video.device}")
        print()

        # Initialize detector
        print("Initializing YOLOv8 detector...")
        detector = YOLOv8Detector(config.inference)

        if detector.is_ready:
            print("✓ Detector initialized successfully")
        else:
            print("⚠ Detector initialization failed (will stream without detections)")

        print()

        # Initialize streamer
        print("Initializing MJPEG streamer...")
        # Use current directory for web files during testing
        web_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'hailo_detector', 'web')
        streamer = MJPEGStreamer(config.stream, web_dir=web_dir)
        streamer.start()

        # Give server time to start
        time.sleep(1)

        local_ip = get_local_ip()
        print(f"✓ MJPEG server started")
        print()
        print("=" * 70)
        print("Stream URLs:")
        print(f"  Web UI:  http://{local_ip}:{config.stream.port}/")
        print(f"  Stream:  http://{local_ip}:{config.stream.port}/stream")
        print(f"  Stats:   http://{local_ip}:{config.stream.port}/api/stats")
        print(f"  Health:  http://{local_ip}:{config.stream.port}/health")
        print("=" * 70)
        print()
        print("Test will run for 60 seconds (or press Ctrl+C to stop)...")
        print()

        # Run for 60 seconds
        class_names = get_coco_class_names()
        start_time = time.time()
        frame_count = 0
        fps_start = time.time()

        stats_interval = 5.0  # Print stats every 5 seconds
        last_stats_time = time.time()

        while not shutdown_requested and (time.time() - start_time) < 60:
            # Check camera
            if not video_capture.is_opened:
                print("Camera disconnected, attempting reconnect...")
                if not video_capture.reconnect():
                    time.sleep(1)
                    continue

            # Capture frame
            frame = video_capture.read()
            if frame is None:
                continue

            # Detect objects
            inference_start = time.time()
            boxes, scores, class_ids = detector.detect(frame)
            inference_time = time.time() - inference_start

            # Draw detections
            annotated = draw_detections(frame, boxes, scores, class_ids, class_names)

            # Update streamer
            streamer.update_frame(annotated)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Update stats
            streamer.update_stats(fps, inference_time, len(boxes))

            # Print stats periodically
            if time.time() - last_stats_time >= stats_interval:
                print(f"Stats: FPS={fps:.1f}, Inference={inference_time*1000:.1f}ms, "
                      f"Detections={len(boxes)}, Frames={frame_count}")
                last_stats_time = time.time()

            # Small delay
            time.sleep(0.001)

        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print()
        print("=" * 70)
        print("Test Complete!")
        print("=" * 70)
        print(f"Total runtime: {total_time:.1f} seconds")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print()

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        print("Cleaning up...")
        if video_capture:
            video_capture.release()
        if detector:
            detector.cleanup()
        if streamer:
            streamer.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
