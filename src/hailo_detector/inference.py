"""
Hailo inference wrapper module for FleeKey Object Detection.
"""

import cv2
import logging
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)


class HailoInference:
    """Hailo-8 inference wrapper using InferVStreams API."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = None
        self.network_group = None
        self.input_shape = None
        self.is_initialized = False
        self.hef = None
        self.input_name = None
        self.output_name = None
        self.network_group_params = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None

        try:
            from hailo_platform import (
                HEF, VDevice, FormatType, HailoStreamInterface,
                ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams
            )
            self.HEF = HEF
            self.VDevice = VDevice
            self.FormatType = FormatType
            self.HailoStreamInterface = HailoStreamInterface
            self.ConfigureParams = ConfigureParams
            self.InferVStreams = InferVStreams
            self.InputVStreamParams = InputVStreamParams
            self.OutputVStreamParams = OutputVStreamParams
        except ImportError as e:
            logger.error(f"Failed to import hailo_platform: {e}")
            raise

    def initialize(self) -> bool:
        """Initialize Hailo device and load model."""
        try:
            logger.info(f"Loading model: {self.model_path}")
            self.hef = self.HEF(self.model_path)

            input_info = self.hef.get_input_vstream_infos()[0]
            output_info = self.hef.get_output_vstream_infos()[0]
            self.input_name = input_info.name
            self.output_name = output_info.name
            self.input_shape = input_info.shape

            logger.info(f"Model loaded - Input: {self.input_shape}, Output: {output_info.shape}")

            self.device = self.VDevice()
            configure_params = self.ConfigureParams.create_from_hef(
                self.hef, interface=self.HailoStreamInterface.PCIe
            )
            self.network_group = self.device.configure(self.hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            self.input_vstreams_params = self.InputVStreamParams.make(
                self.network_group, format_type=self.FormatType.UINT8
            )
            self.output_vstreams_params = self.OutputVStreamParams.make(
                self.network_group, format_type=self.FormatType.FLOAT32
            )

            self.is_initialized = True
            logger.info("Hailo inference initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Hailo inference: {e}")
            self.is_initialized = False
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB UINT8 for inference."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    def infer(self, image: np.ndarray) -> Optional[List]:
        """
        Run inference and return class-separated detections.

        Returns:
            List of 80 class arrays, each with detections [y1,x1,y2,x2,score]
        """
        if not self.is_initialized:
            return None

        try:
            input_data = np.expand_dims(image, axis=0)

            with self.InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params
            ) as pipeline:
                with self.network_group.activate(self.network_group_params):
                    outputs = pipeline.infer({self.input_name: input_data})

            return outputs[self.output_name][0]

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None

    def cleanup(self):
        """Release Hailo resources."""
        self.network_group = None
        self.hef = None
        self.device = None
        self.is_initialized = False

    def __del__(self):
        self.cleanup()
