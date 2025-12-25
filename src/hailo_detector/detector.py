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
        self.detection_count = 0  # For debug logging throttling

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
            import time
            t_start = time.perf_counter()

            # Preprocess image with letterboxing
            letterboxed, scale, pad = letterbox_image(image, self.config.input_size)
            t_letterbox = time.perf_counter()

            # Preprocess for inference (normalize, convert to RGB)
            preprocessed = self.inference_engine.preprocess(letterboxed)
            t_preprocess = time.perf_counter()

            # Run inference
            outputs = self.inference_engine.infer(preprocessed)
            t_infer = time.perf_counter()

            if outputs is None:
                logger.warning("Inference returned None")
                return np.array([]), np.array([]), np.array([])

            # Post-process outputs
            boxes, scores, class_ids = self._postprocess(
                outputs, scale, pad, image.shape[:2]
            )
            t_postprocess = time.perf_counter()

            # Log timing every 30 frames
            if self.detection_count % 30 == 1:
                logger.info(f"=== TIMING BREAKDOWN (ms) ===")
                logger.info(f"Letterbox:    {(t_letterbox - t_start) * 1000:.2f}ms")
                logger.info(f"Preprocess:   {(t_preprocess - t_letterbox) * 1000:.2f}ms")
                logger.info(f"Inference:    {(t_infer - t_preprocess) * 1000:.2f}ms")
                logger.info(f"Postprocess:  {(t_postprocess - t_infer) * 1000:.2f}ms")
                logger.info(f"TOTAL:        {(t_postprocess - t_start) * 1000:.2f}ms")

            return boxes, scores, class_ids

        except Exception as e:
            import traceback
            logger.error(f"Detection failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        try:
            # Get the output tensor
            output_key = list(outputs.keys())[0]
            output = outputs[output_key]

            # Track detection count for logging
            self.detection_count += 1

            # Debug: Print raw output info ONLY for first frame
            should_debug = (self.detection_count == 1)
            if should_debug:
                logger.info(f"=== DEBUG OUTPUT INFO (Frame {self.detection_count}) ===")
                logger.info(f"Output key: {output_key}")
                logger.info(f"RAW Output shape: {output.shape}")

            # Handle Hailo NMS postprocessed output
            # Standard format: [num_classes, max_detections, values_per_detection]
            # Values: [y_min, x_min, y_max, x_max, score]
            if isinstance(output, list):
                output = np.array(output)

            num_classes = 80
            values_per_detection = 5
            max_detections_per_class = 100
            expected_size = num_classes * values_per_detection * max_detections_per_class
            values_per_class_with_count = 1 + (max_detections_per_class * values_per_detection)  # 501
            expected_size_with_counts = num_classes * values_per_class_with_count  # 40,080

            # Handle Hailo NMS format with detection counts
            if output.ndim == 1 and output.size == expected_size_with_counts:
                # Format: Each class has [count, det0_val0, det0_val1, ..., det99_val4]
                # Shape: (40080,) -> (80, 501)
                if should_debug:
                    logger.info(f"✓ Hailo NMS format detected: (40080,) with detection counts")

                # Reshape to (80, 501)
                output = output.reshape(num_classes, values_per_class_with_count)

                # Extract detection counts (first value per class) - for debugging
                if should_debug:
                    detection_counts = output[:, 0].astype(int)
                    non_zero_classes = [(i, self.class_names[i], int(detection_counts[i])) for i in range(num_classes) if detection_counts[i] > 0]
                    logger.info(f"Classes with detections: {non_zero_classes}")

                # Skip the first value (count) and take the remaining 500 values
                # Shape: (80, 500) then reshape to (80, 100, 5)
                output = output[:, 1:].reshape(num_classes, max_detections_per_class, values_per_detection)

            elif output.shape == (num_classes, max_detections_per_class, values_per_detection):
                # Already correct shape (80, 100, 5)
                if should_debug:
                    logger.info("✓ Output already in correct shape (80, 100, 5)")
            elif output.shape == (num_classes, values_per_detection, max_detections_per_class):
                # Shape is (80, 5, 100) - need to transpose
                logger.warning("⚠ Output shape is (80, 5, 100), transposing to (80, 100, 5)")
                output = output.transpose(0, 2, 1)  # (80, 5, 100) -> (80, 100, 5)
            elif output.ndim == 1 and output.size == expected_size:
                # Flattened output without counts - reshape to (80, 100, 5)
                if should_debug:
                    logger.info(f"Reshaping flattened output {output.shape} -> (80, 100, 5)")
                output = output.reshape(num_classes, max_detections_per_class, values_per_detection)
            elif output.ndim == 4 and output.shape[0] == 1:
                # Batch dimension present - remove it
                logger.info(f"Removing batch dimension: {output.shape} -> {output[0].shape}")
                output = output[0]
                # Recursively check shape again
                if output.shape == (num_classes, values_per_detection, max_detections_per_class):
                    logger.warning("⚠ After removing batch, transposing (80, 5, 100) -> (80, 100, 5)")
                    output = output.transpose(0, 2, 1)
            else:
                logger.warning(f"⚠ Unexpected output shape: {output.shape}, attempting reshape")
                # Try flattening and reshaping as last resort
                if output.size >= expected_size:
                    output = output.flatten()[:expected_size].reshape(num_classes, max_detections_per_class, values_per_detection)
                else:
                    raise ValueError(f"Output size {output.size} is less than expected {expected_size}")

            if should_debug:
                logger.info(f"FINAL output shape: {output.shape}")

            # Vectorized postprocessing - 100x faster than nested loops!
            # Reshape to (N, 5) where N = 80*100 = 8000
            detections_flat = output.reshape(-1, 5)  # (8000, 5)

            # Extract all fields at once
            y_mins = detections_flat[:, 0]
            x_mins = detections_flat[:, 1]
            y_maxs = detections_flat[:, 2]
            x_maxs = detections_flat[:, 3]
            scores = detections_flat[:, 4]

            # Create class IDs: [0,0,0...0 (100x), 1,1,1...1 (100x), ..., 79,79,79...79 (100x)]
            class_ids_all = np.repeat(np.arange(num_classes), max_detections_per_class)

            # Vectorized filtering - all conditions at once
            valid_mask = (
                (scores > self.config.confidence_threshold) &  # Above threshold
                (scores > 0) &  # Valid score
                (y_mins >= 0) & (y_mins <= 1) &  # Y in bounds
                (y_maxs >= 0) & (y_maxs <= 1) &
                (x_mins >= 0) & (x_mins <= 1) &  # X in bounds
                (x_maxs >= 0) & (x_maxs <= 1) &
                (x_maxs > x_mins) &  # Non-degenerate width
                (y_maxs > y_mins)    # Non-degenerate height
            )

            # Apply mask - extract only valid detections
            valid_scores = scores[valid_mask]
            valid_y_mins = y_mins[valid_mask]
            valid_x_mins = x_mins[valid_mask]
            valid_y_maxs = y_maxs[valid_mask]
            valid_x_maxs = x_maxs[valid_mask]
            valid_class_ids = class_ids_all[valid_mask]

            # Debug logging for first frame only
            if should_debug and len(valid_scores) > 0:
                logger.info(f"Frame 1: Found {len(valid_scores)} valid detections")
                for i in range(min(2, len(valid_scores))):
                    logger.info(f"  - class={valid_class_ids[i]}({self.class_names[valid_class_ids[i]]}) score={valid_scores[i]:.3f}")

            if len(valid_scores) == 0:
                # Debug: track max scores per class
                if self.detection_count % 30 == 1:
                    max_scores_per_class = output[:, :, 4].max(axis=1)  # (80,)
                    top_classes = [(i, self.class_names[i], f'{max_scores_per_class[i]:.3f}')
                                 for i in range(num_classes) if max_scores_per_class[i] > 0]
                    top_classes.sort(key=lambda x: float(x[2]), reverse=True)
                    logger.info(f"No valid detections. Top scores: {top_classes[:5]}")
                return np.array([]), np.array([]), np.array([])

            # Vectorized coordinate scaling to model input size
            model_input_size = self.config.input_size
            boxes_xyxy = np.stack([
                valid_x_mins * model_input_size,  # x1
                valid_y_mins * model_input_size,  # y1
                valid_x_maxs * model_input_size,  # x2
                valid_y_maxs * model_input_size   # y2
            ], axis=1)

            # Minimal logging every 100 frames
            if self.detection_count % 100 == 1 and len(valid_scores) > 0:
                unique_classes = np.unique(valid_class_ids)
                class_names_detected = [self.class_names[int(c)] for c in unique_classes]
                logger.info(f"Frame {self.detection_count}: {len(valid_scores)} detections ({', '.join(class_names_detected[:5])})")

            # NMS already done by Hailo, skip additional NMS
            scores = valid_scores.astype(np.float32)
            class_ids = valid_class_ids.astype(np.int32)

            # Scale boxes back to original image coordinates
            boxes_xyxy = scale_boxes(boxes_xyxy, scale, pad)

            # Clip boxes to image boundaries
            h, w = original_shape
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

            return boxes_xyxy, scores, class_ids

        except Exception as e:
            import traceback
            logger.error(f"Postprocessing failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([]), np.array([]), np.array([])

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
