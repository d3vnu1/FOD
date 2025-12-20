"""
YOLOv8 detection and post-processing module.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional
from .config import InferenceConfig
from .inference import HailoInference
from .utils import letterbox_image, scale_boxes, get_coco_class_names


logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    YOLOv8 object detector using Hailo-8L acceleration.
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize YOLOv8 detector.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.class_names = get_coco_class_names()
        self.inference_engine: Optional[HailoInference] = None
        self.is_ready = False

        # Try to initialize Hailo inference
        try:
            self.inference_engine = HailoInference(config.model_path)
            self.is_ready = self.inference_engine.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            self.is_ready = False

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in image.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (boxes, scores, class_ids)
            - boxes: Array of [x1, y1, x2, y2] in original image coordinates
            - scores: Confidence scores
            - class_ids: Class IDs
        """
        if not self.is_ready or self.inference_engine is None:
            logger.warning("Detector not ready, returning empty detections")
            return np.array([]), np.array([]), np.array([])

        try:
            # Preprocess image with letterboxing
            letterboxed, scale, pad = letterbox_image(image, self.config.input_size)

            # Preprocess for inference (normalize, convert to RGB)
            preprocessed = self.inference_engine.preprocess(letterboxed)

            # Run inference
            outputs = self.inference_engine.infer(preprocessed)

            if outputs is None:
                logger.warning("Inference returned None")
                return np.array([]), np.array([]), np.array([])

            # Post-process outputs
            boxes, scores, class_ids = self._postprocess(
                outputs, scale, pad, image.shape[:2]
            )

            return boxes, scores, class_ids

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return np.array([]), np.array([]), np.array([])

    def _postprocess(self, outputs: dict, scale: float, pad: Tuple[int, int],
                     original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process YOLOv8 outputs.

        Args:
            outputs: Raw model outputs
            scale: Scale factor from letterboxing
            pad: Padding from letterboxing
            original_shape: Original image shape (height, width)

        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        # Get the output tensor
        # YOLOv8 output format is typically (1, 84, 8400) for COCO
        # where 84 = 4 (bbox) + 80 (classes)
        output_key = list(outputs.keys())[0]
        output = outputs[output_key]

        # Transpose to (1, 8400, 84) for easier processing
        if output.shape[-1] != 84:
            output = output.transpose(0, 2, 1)

        # Remove batch dimension
        output = output[0]  # Shape: (8400, 84)

        # Split into boxes and class scores
        boxes = output[:, :4]  # (8400, 4) - [cx, cy, w, h]
        class_scores = output[:, 4:]  # (8400, 80)

        # Get max class score and class ID for each detection
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= self.config.confidence_threshold
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Apply NMS
        indices = self._nms(boxes_xyxy, scores, self.config.nms_threshold)
        boxes_xyxy = boxes_xyxy[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]

        # Scale boxes back to original image coordinates
        boxes_xyxy = scale_boxes(boxes_xyxy, scale, pad)

        # Clip boxes to image boundaries
        h, w = original_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

        return boxes_xyxy, scores, class_ids

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Non-Maximum Suppression using OpenCV.

        Args:
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            scores: Confidence scores
            iou_threshold: IOU threshold for NMS

        Returns:
            Indices of boxes to keep
        """
        # Convert to format expected by cv2.dnn.NMSBoxes [x, y, w, h]
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.config.confidence_threshold,
            iou_threshold
        )

        if len(indices) > 0:
            return indices.flatten()
        else:
            return np.array([])

    def cleanup(self):
        """Release detector resources."""
        if self.inference_engine is not None:
            self.inference_engine.cleanup()
        self.is_ready = False

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
