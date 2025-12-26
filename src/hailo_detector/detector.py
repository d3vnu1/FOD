"""
YOLOv8 detection module for FleeKey Object Detection.
"""

import logging
import numpy as np
from typing import Tuple, Optional, List
from .config import InferenceConfig
from .inference import HailoInference
from .utils import letterbox_image, scale_boxes, get_coco_class_names

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """YOLOv8 object detector using Hailo-8 acceleration."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.class_names = get_coco_class_names()
        self.inference_engine: Optional[HailoInference] = None
        self.is_ready = False
        self.frame_count = 0

        try:
            self.inference_engine = HailoInference(config.model_path)
            self.is_ready = self.inference_engine.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            self.is_ready = False

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in image.

        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        if not self.is_ready or self.inference_engine is None:
            return np.array([]), np.array([]), np.array([])

        try:
            letterboxed, scale, pad = letterbox_image(image, self.config.input_size)
            preprocessed = self.inference_engine.preprocess(letterboxed)
            outputs = self.inference_engine.infer(preprocessed)

            if outputs is None:
                return np.array([]), np.array([]), np.array([])

            return self._postprocess(outputs, scale, pad, image.shape[:2])

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return np.array([]), np.array([]), np.array([])

    def _postprocess(self, outputs: List, scale: float, pad: Tuple[int, int],
                     original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post-process class-separated detections from Hailo NMS."""
        self.frame_count += 1
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for class_id, class_detections in enumerate(outputs):
            if class_detections is None:
                continue

            dets = np.array(class_detections)
            if dets.size == 0:
                continue

            if dets.ndim == 1:
                dets = dets.reshape(1, -1)

            if dets.shape[1] != 5:
                continue

            for det in dets:
                y1, x1, y2, x2, score = det

                if score < self.config.confidence_threshold:
                    continue

                if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                    continue

                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                if x_max <= x_min or y_max <= y_min:
                    continue

                model_size = self.config.input_size
                box = [
                    x_min * model_size,
                    y_min * model_size,
                    x_max * model_size,
                    y_max * model_size
                ]

                all_boxes.append(box)
                all_scores.append(float(score))
                all_class_ids.append(class_id)

        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        class_ids = np.array(all_class_ids, dtype=np.int32)

        boxes = scale_boxes(boxes, scale, pad)

        h, w = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        return boxes, scores, class_ids

    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def cleanup(self):
        """Release detector resources."""
        if self.inference_engine is not None:
            self.inference_engine.cleanup()
        self.is_ready = False

    def __del__(self):
        self.cleanup()
