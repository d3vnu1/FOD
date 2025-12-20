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

        # Try to import hailo_platform
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoStreamInterface,
                InferVStreams,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType
            )
            self.HEF = HEF
            self.VDevice = VDevice
            self.HailoStreamInterface = HailoStreamInterface
            self.InferVStreams = InferVStreams
            self.ConfigureParams = ConfigureParams
            self.InputVStreamParams = InputVStreamParams
            self.OutputVStreamParams = OutputVStreamParams
            self.FormatType = FormatType
            logger.info("hailo_platform module imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import hailo_platform: {e}")
            logger.error("Make sure HailoRT is installed on your system")
            raise

    def initialize(self) -> bool:
        """
        Initialize Hailo device and load model.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading HEF model: {self.model_path}")

            # Load HEF
            hef = self.HEF(self.model_path)

            # Create VDevice
            params = self.VDevice.create_params()
            self.device = self.VDevice(params)

            logger.info("Hailo device created successfully")

            # Configure network group
            network_group = self.device.configure(hef)[0]
            self.network_group = network_group

            # Get input/output parameters
            input_vstream_params = self.InputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=self.FormatType.FLOAT32
            )
            output_vstream_params = self.OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=self.FormatType.FLOAT32
            )

            # Get input shape
            self.input_shape = input_vstream_params[0].shape

            logger.info(f"Model input shape: {self.input_shape}")

            # Create inference pipeline
            self.infer_pipeline = self.InferVStreams(
                network_group, input_vstream_params, output_vstream_params
            )

            self.is_initialized = True
            logger.info("Hailo inference initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Hailo inference: {e}")
            self.is_initialized = False
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image in BGR format (640x640x3)

        Returns:
            Preprocessed image ready for inference
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0

        # Add batch dimension if needed
        if len(image_normalized.shape) == 3:
            image_normalized = np.expand_dims(image_normalized, axis=0)

        return image_normalized

    def infer(self, image: np.ndarray) -> Optional[dict]:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image (1, H, W, 3) in RGB format, float32, [0, 1]

        Returns:
            Dictionary of output tensors, or None if failed
        """
        if not self.is_initialized:
            logger.error("Hailo inference not initialized")
            return None

        try:
            # Prepare input dictionary
            input_data = {list(self.infer_pipeline.input_vstreams.keys())[0]: image}

            # Run inference
            with self.infer_pipeline:
                output_dict = self.infer_pipeline.infer(input_data)

            return output_dict

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None

    def cleanup(self):
        """Release Hailo resources."""
        try:
            if self.network_group is not None:
                logger.info("Releasing Hailo network group")
                self.network_group = None

            if self.device is not None:
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
