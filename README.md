# FleeKey Object Detection

Real-time object detection service for **Raspberry Pi CM5** with **Hailo-8 AI accelerator** (26 TOPS). Captures video from USB camera, runs YOLOv8 inference on Hailo-8, and streams annotated video via HTTP MJPEG.

## Features

- **Real-time Detection**: YOLOv8s model on Hailo-8 at 30+ FPS
- **80 COCO Classes**: Detects person, car, chair, bottle, and 76 more object types
- **Headless Operation**: No display required - view via web browser
- **System Monitoring**: CPU load, CPU temperature, Hailo temperature
- **MJPEG Streaming**: View live annotated video from any device
- **REST API**: Statistics and system monitoring endpoints
- **Systemd Service**: Automatic startup and crash recovery

## Hardware Requirements

- **SBC**: Raspberry Pi Compute Module 5 (ARM64)
- **AI Accelerator**: Hailo-8 (26 TOPS) via PCIe
- **Camera**: V4L2-compatible USB camera
- **OS**: Raspberry Pi OS Bookworm 64-bit

## Installation

### Prerequisites

Install HailoRT from Hailo's official repository:

```bash
# Follow official Hailo RPi5 installation guide
```

### Build and Install

```bash
git clone <repository-url>
cd FOD
make build
make install
```

## Configuration

Edit `/etc/hailo-detector/config.yaml`:

```yaml
video:
  device: "/dev/video0"
  width: 1920
  height: 1080
  fps: 30

inference:
  model_path: "/opt/hailo-detector/models/yolov8s_h8.hef"
  confidence_threshold: 0.5
  input_size: 640

stream:
  host: "0.0.0.0"
  port: 8080
  jpeg_quality: 80

logging:
  level: "INFO"
```

Restart service after changes:

```bash
sudo systemctl restart hailo-detector
```

## Usage

### Web Interface

Navigate to `http://<device-ip>:8080/` to view:
- Live annotated video stream
- Real-time FPS and inference stats
- Detection count

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Web UI |
| `/stream` | MJPEG video stream |
| `/api/stats` | Detection statistics (JSON) |
| `/api/system` | System monitoring (JSON) |
| `/health` | Health check |

### System Monitoring API

`GET /api/system` returns:

```json
{
  "cpu": {
    "load_1m": 0.5,
    "load_5m": 0.4,
    "load_15m": 0.3,
    "temperature": 45.2
  },
  "hailo": {
    "temperature": 42.0
  },
  "memory": {
    "total_mb": 8192,
    "used_mb": 1024,
    "available_mb": 7168
  }
}
```

### Service Management

```bash
sudo systemctl start hailo-detector
sudo systemctl stop hailo-detector
sudo systemctl restart hailo-detector
systemctl status hailo-detector
journalctl -u hailo-detector -f
```

## Project Structure

```
FOD/
├── src/hailo_detector/
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── config.py        # Configuration
│   ├── capture.py       # Video capture
│   ├── inference.py     # Hailo inference (InferVStreams API)
│   ├── detector.py      # YOLOv8 detection
│   ├── streamer.py      # MJPEG server
│   ├── monitor.py       # System monitoring
│   └── utils.py         # Utilities
├── models/
│   └── yolov8s_h8_v2.13.hef
├── config/
│   └── config.yaml.example
├── debian/              # Packaging
├── Makefile
└── pyproject.toml
```

## Performance

On Raspberry Pi CM5 with Hailo-8:

- **FPS**: 30+ (1920x1080 capture, 640x640 inference)
- **Inference**: ~20-30ms per frame
- **Latency**: <50ms capture to output
- **CPU Usage**: ~15-25%

## Technical Notes

### InferVStreams API

This project uses the HailoRT `InferVStreams` API for proper class-separated NMS output. The model returns 80 arrays (one per COCO class) with detections in `[y1, x1, y2, x2, score]` format.

### Model

Uses `yolov8s_h8_v2.13.hef` from Hailo Model Zoo, compiled for Hailo-8 with built-in NMS postprocessing.

## License

MIT License

## Credits

- **Hailo AI**: Hailo-8 accelerator and HailoRT
- **Ultralytics**: YOLOv8 model architecture
- **OpenCV**: Image processing
