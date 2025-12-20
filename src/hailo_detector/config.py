"""
Configuration management using Pydantic for validation and type checking.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class VideoConfig(BaseModel):
    """Video capture configuration."""
    device: str = Field(default="/dev/video0", description="Video device path")
    width: int = Field(default=1920, ge=320, le=3840, description="Capture width")
    height: int = Field(default=1080, ge=240, le=2160, description="Capture height")
    fps: int = Field(default=30, ge=1, le=60, description="Capture FPS")


class InferenceConfig(BaseModel):
    """Inference configuration."""
    model_path: str = Field(
        default="/opt/hailo-detector/models/yolov8n.hef",
        description="Path to HEF model file"
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold"
    )
    nms_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="NMS IOU threshold"
    )
    input_size: int = Field(
        default=640, ge=320, le=1280, description="Model input size"
    )


class StreamConfig(BaseModel):
    """Streaming server configuration."""
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8080, ge=1024, le=65535, description="Server port")
    jpeg_quality: int = Field(
        default=80, ge=1, le=100, description="JPEG compression quality"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Logging level")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of {valid_levels}")
        return v


class Config(BaseModel):
    """Main configuration class."""
    video: VideoConfig = Field(default_factory=VideoConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with sensible defaults.

    Args:
        config_path: Path to configuration file. If None, uses default location.

    Returns:
        Config object with loaded settings.
    """
    # Default config path
    if config_path is None:
        config_path = "/etc/hailo-detector/config.yaml"

    # If config file doesn't exist, use defaults
    if not os.path.exists(config_path):
        return Config()

    # Load YAML configuration
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle empty file
        if config_dict is None:
            return Config()

        return Config(**config_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")


def save_example_config(output_path: str) -> None:
    """
    Save an example configuration file with comments.

    Args:
        output_path: Where to save the example config.
    """
    example_yaml = """# Video capture settings
video:
  device: "/dev/video0"  # V4L2 device path
  width: 1920            # Capture width in pixels
  height: 1080           # Capture height in pixels
  fps: 30                # Frames per second

# Inference settings
inference:
  model_path: "/opt/hailo-detector/models/yolov8n.hef"  # Path to HEF model
  confidence_threshold: 0.5  # Minimum confidence for detections (0.0-1.0)
  nms_threshold: 0.45        # Non-maximum suppression threshold (0.0-1.0)
  input_size: 640            # Model input size (usually 640 for YOLOv8)

# Streaming settings
stream:
  host: "0.0.0.0"      # Bind to all interfaces
  port: 8080           # HTTP server port
  jpeg_quality: 80     # JPEG compression quality (1-100)

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(example_yaml)
