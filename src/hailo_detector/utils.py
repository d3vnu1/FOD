"""
Utility functions for image processing and general helpers.
"""

import cv2
import numpy as np
from typing import Tuple


def letterbox_image(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing to maintain aspect ratio.

    Args:
        image: Input image (BGR format)
        target_size: Target square size (e.g., 640)

    Returns:
        Tuple of (letterboxed_image, scale_factor, (pad_x, pad_y))
    """
    h, w = image.shape[:2]

    # Calculate scale to fit image within target size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create letterboxed image (filled with gray)
    letterboxed = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # Calculate padding
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    # Place resized image in center
    letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return letterboxed, scale, (pad_x, pad_y)


def scale_boxes(boxes: np.ndarray, scale: float, pad: Tuple[int, int]) -> np.ndarray:
    """
    Scale bounding boxes back to original image coordinates.

    Args:
        boxes: Array of boxes in format [x1, y1, x2, y2]
        scale: Scale factor from letterboxing
        pad: Padding from letterboxing (pad_x, pad_y)

    Returns:
        Scaled boxes in original image coordinates
    """
    pad_x, pad_y = pad

    # Remove padding
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y

    # Scale back to original size
    boxes = boxes / scale

    return boxes


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                    class_ids: np.ndarray, class_names: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: Input image (BGR format)
        boxes: Bounding boxes in format [x1, y1, x2, y2]
        scores: Confidence scores
        class_ids: Class IDs
        class_names: List of class names

    Returns:
        Annotated image
    """
    annotated = image.copy()

    # Colors for different classes (using OpenCV BGR format)
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
    ]

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)

        # Ensure coordinates are within image bounds
        h, w = annotated.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Get color for this class
        color = colors[int(class_id) % len(colors)]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        label = f"{class_names[int(class_id)]}: {score:.2f}"

        # Calculate label size and position
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated


def get_coco_class_names() -> list:
    """
    Get COCO dataset class names (80 classes used by YOLOv8).

    Returns:
        List of class names
    """
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
