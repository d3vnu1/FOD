"""
Flask-based MJPEG streaming server with web UI.
"""

import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional
from flask import Flask, Response, render_template_string, jsonify
from .config import StreamConfig
from .monitor import SystemMonitor


logger = logging.getLogger(__name__)


class MJPEGStreamer:
    """
    MJPEG streaming server using Flask.
    """

    def __init__(self, config: StreamConfig, web_dir: str = "/opt/hailo-detector/web"):
        """
        Initialize MJPEG streamer.

        Args:
            config: Stream configuration
            web_dir: Directory containing web UI files
        """
        self.config = config
        self.web_dir = web_dir
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)

        # Thread-safe frame buffer
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Statistics
        self.stats = {
            'fps': 0.0,
            'inference_time': 0.0,
            'num_detections': 0,
            'uptime': 0,
            'total_frames': 0
        }
        self.stats_lock = threading.Lock()
        self.start_time = time.time()

        # System monitor
        self.system_monitor = SystemMonitor()

        # Setup routes
        self._setup_routes()

        # Server thread
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Serve web UI."""
            try:
                index_path = Path(self.web_dir) / 'index.html'
                if index_path.exists():
                    with open(index_path, 'r') as f:
                        return f.read()
                else:
                    # Fallback minimal UI if file doesn't exist
                    return self._get_minimal_ui()
            except Exception as e:
                logger.error(f"Error serving index: {e}")
                return self._get_minimal_ui()

        @self.app.route('/stream')
        def stream():
            """MJPEG stream endpoint."""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/health')
        def health():
            """Health check endpoint."""
            with self.stats_lock:
                uptime = int(time.time() - self.start_time)
                return jsonify({
                    'status': 'running',
                    'uptime': uptime,
                    'fps': round(self.stats['fps'], 2)
                })

        @self.app.route('/api/stats')
        def stats():
            """Statistics endpoint."""
            with self.stats_lock:
                return jsonify({
                    'fps': round(self.stats['fps'], 2),
                    'inference_time': round(self.stats['inference_time'], 3),
                    'num_detections': self.stats['num_detections'],
                    'uptime': int(time.time() - self.start_time),
                    'total_frames': self.stats['total_frames']
                })

        @self.app.route('/api/system')
        def system():
            """System monitoring endpoint."""
            return jsonify(self.system_monitor.get_all_stats())

    def _get_minimal_ui(self) -> str:
        """Get minimal fallback UI."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>FleeKey Object Detection</title></head>
        <body>
            <h1>FleeKey Object Detection</h1>
            <img src="/stream" style="max-width: 100%;">
        </body>
        </html>
        """

    def _generate_frames(self):
        """
        Generator for MJPEG frames.

        Yields:
            MJPEG frame data
        """
        while True:
            with self.frame_lock:
                if self.current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode(
                        '.jpg',
                        self.current_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                    )

                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Small delay to prevent busy waiting
            time.sleep(0.01)

    def update_frame(self, frame: np.ndarray):
        """
        Update the current frame to be streamed.

        Args:
            frame: New frame (BGR format)
        """
        with self.frame_lock:
            self.current_frame = frame.copy()

    def update_stats(self, fps: float, inference_time: float, num_detections: int):
        """
        Update statistics.

        Args:
            fps: Current FPS
            inference_time: Inference time in seconds
            num_detections: Number of detections in last frame
        """
        with self.stats_lock:
            self.stats['fps'] = fps
            self.stats['inference_time'] = inference_time
            self.stats['num_detections'] = num_detections
            self.stats['total_frames'] += 1

    def start(self):
        """Start the streaming server in a separate thread."""
        if self.is_running:
            logger.warning("Streamer already running")
            return

        logger.info(f"Starting MJPEG server on {self.config.host}:{self.config.port}")

        def run_server():
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                threaded=True,
                debug=False,
                use_reloader=False
            )

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True

        logger.info("MJPEG server started")

    def stop(self):
        """Stop the streaming server."""
        self.is_running = False
        logger.info("MJPEG server stopped")
