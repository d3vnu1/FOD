"""
Video capture module using OpenCV with V4L2 backend.
"""

import cv2
import time
import logging
import numpy as np
from typing import Optional
from .config import VideoConfig


logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Video capture class with automatic reconnection and error handling.
    """

    def __init__(self, config: VideoConfig):
        """
        Initialize video capture.

        Args:
            config: Video configuration
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.last_frame: Optional[np.ndarray] = None
        self.frame_count = 0

    def open(self) -> bool:
        """
        Open video capture device.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Opening video device: {self.config.device}")

            # Detect if source is a video file or camera device
            is_video_file = self.config.device.endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))

            if is_video_file:
                # Use default backend for video files
                logger.info(f"Detected video file: {self.config.device}")
                self.cap = cv2.VideoCapture(self.config.device)
            else:
                # Use V4L2 backend for camera devices
                self.cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                logger.error(f"Failed to open {self.config.device}")
                return False

            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )

            self.is_opened = True
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            self.is_opened = False
            return False

    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.

        Returns:
            Frame as numpy array (BGR), or None if failed
        """
        if not self.is_opened or self.cap is None:
            return None

        try:
            ret, frame = self.cap.read()

            if ret and frame is not None:
                self.last_frame = frame
                self.frame_count += 1
                return frame
            else:
                logger.warning("Failed to read frame from camera")
                self.is_opened = False
                return None

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            self.is_opened = False
            return None

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            logger.info("Releasing video capture")
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def reconnect(self, retry_interval: float = 5.0) -> bool:
        """
        Try to reconnect to camera.

        Args:
            retry_interval: Seconds to wait between retries

        Returns:
            True if reconnected successfully
        """
        logger.info(f"Attempting to reconnect to {self.config.device}...")

        # Release existing capture
        self.release()

        # Wait before retry
        time.sleep(retry_interval)

        # Try to reopen
        return self.open()

    def get_last_frame(self) -> Optional[np.ndarray]:
        """
        Get the last successfully captured frame.

        Returns:
            Last frame or None
        """
        return self.last_frame

    def __del__(self):
        """Cleanup on deletion."""
        self.release()
