"""
Hailo-8L inference wrapper module.
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class HailoInference:
    """
    Wrapper for Hailo-8L inference using HailoRT.
    """

    def __init__(self, model_path: str):
        """
        Initialize Hailo inference.

        Args:
            model_path: Path to HEF model file
        """
        self.model_path = model_path
        self.device = None
        self.network_group = None
        self.input_vstream = None
        self.output_vstreams = None
        self.input_shape = None
        self.is_initialized = False
        self.inference_count = 0  # For debug logging throttling

        # Pre-allocated buffers for fast inference
        self.configured_infer_model = None
        self.output_buffers = None
        self.bindings = None
        self.output_names = None

        # Try to import hailo_platform
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                FormatType,
                HailoSchedulingAlgorithm
            )
            self.HEF = HEF
            self.VDevice = VDevice
            self.FormatType = FormatType
            self.HailoSchedulingAlgorithm = HailoSchedulingAlgorithm
            logger.info("hailo_platform module imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import hailo_platform: {e}")
            logger.error("Make sure HailoRT is installed on your system")
            raise

    def initialize(self) -> bool:
        """
        Initialize Hailo device and load model using create_infer_model API.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading HEF model: {self.model_path}")

            # Create VDevice with scheduling params
            params = self.VDevice.create_params()
            params.scheduling_algorithm = self.HailoSchedulingAlgorithm.ROUND_ROBIN
            self.device = self.VDevice(params)
            logger.info("Hailo device created successfully")

            # Create infer model (simpler API than InferVStreams)
            self.infer_model = self.device.create_infer_model(self.model_path)
            logger.info("Infer model created successfully")

            # Set batch size
            self.infer_model.set_batch_size(1)
            logger.info("Batch size set to 1")

            # Set input format to UINT8 (as per working example)
            self.infer_model.input().set_format_type(self.FormatType.UINT8)
            logger.info("Input format set to UINT8")

            # Get input shape (shape is a property, not a method)
            self.input_shape = self.infer_model.input().shape
            logger.info(f"Model input shape: {self.input_shape}")

            # Get output info (output_names is a property, not a method)
            self.output_names = self.infer_model.output_names
            logger.info(f"Output names: {self.output_names}")

            # Pre-configure model and allocate buffers for fast inference
            logger.info("Pre-allocating buffers for optimized inference...")
            self.configured_infer_model = self.infer_model.configure()

            # Allocate output buffers once
            self.output_buffers = {}
            for name in self.output_names:
                output_shape = self.infer_model.output(name).shape
                output_size = np.prod(output_shape)
                self.output_buffers[name] = np.empty(output_size, dtype=np.float32)
                logger.info(f"Allocated buffer for {name}: shape={output_shape}, size={output_size}")

            # Create bindings once
            self.bindings = self.configured_infer_model.create_bindings()

            # Set output buffers once
            for name in self.output_names:
                self.bindings.output(name).set_buffer(self.output_buffers[name])

            logger.info("Buffers pre-allocated successfully")

            self.is_initialized = True
            logger.info("Hailo inference initialized successfully")
            return True

        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize Hailo inference: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.is_initialized = False
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image in BGR format (640x640x3)

        Returns:
            Preprocessed image ready for inference (UINT8, RGB)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure UINT8 format [0-255]
        image_uint8 = image_rgb.astype(np.uint8)

        return image_uint8

    def infer(self, image: np.ndarray) -> Optional[dict]:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image (H, W, 3) in RGB format, uint8, [0-255]

        Returns:
            Dictionary of output tensors, or None if failed
        """
        if not self.is_initialized:
            logger.error("Hailo inference not initialized")
            return None

        try:
            # Use pre-allocated buffers and bindings for fast inference
            # Set input tensor (only thing that changes each frame)
            self.bindings.input().set_buffer(image)

            # Run inference with timeout (5000ms = 5 seconds)
            self.configured_infer_model.run([self.bindings], timeout=5000)

            # Get output from pre-allocated buffers and reshape
            results = {}
            for name in self.output_names:
                output_shape = self.infer_model.output(name).shape
                # Return view of buffer (no copy needed - postprocessing uses it immediately)
                results[name] = self.output_buffers[name].reshape(output_shape)

            return results

        except Exception as e:
            import traceback
            logger.error(f"Inference failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def cleanup(self):
        """Release Hailo resources."""
        try:
            if hasattr(self, 'configured_infer_model') and self.configured_infer_model is not None:
                logger.info("Releasing configured infer model")
                self.configured_infer_model = None

            if hasattr(self, 'bindings') and self.bindings is not None:
                logger.info("Releasing bindings")
                self.bindings = None

            if hasattr(self, 'output_buffers') and self.output_buffers is not None:
                logger.info("Releasing output buffers")
                self.output_buffers = None

            if hasattr(self, 'infer_model') and self.infer_model is not None:
                logger.info("Releasing Hailo infer model")
                self.infer_model = None

            if hasattr(self, 'device') and self.device is not None:
                logger.info("Releasing Hailo device")
                self.device = None

            self.is_initialized = False

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


# Import cv2 here for preprocess method
import cv2
