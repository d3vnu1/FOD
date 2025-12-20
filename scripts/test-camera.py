#!/usr/bin/env python3
"""
Camera test utility to verify camera connectivity and Hailo device access.
"""

import sys
import cv2


def test_camera(device_path="/dev/video0"):
    """
    Test camera connectivity and capture capabilities.

    Args:
        device_path: Path to video device
    """
    print("=" * 70)
    print("Hailo Detector - Camera Test Utility")
    print("=" * 70)
    print()

    # Test 1: Check if device exists
    print(f"[1/5] Checking if device exists: {device_path}")
    import os
    if os.path.exists(device_path):
        print(f"  ✓ Device found: {device_path}")
    else:
        print(f"  ✗ Device not found: {device_path}")
        print("  Available video devices:")
        for i in range(10):
            dev = f"/dev/video{i}"
            if os.path.exists(dev):
                print(f"    - {dev}")
        return False
    print()

    # Test 2: Open camera
    print(f"[2/5] Opening camera with V4L2 backend...")
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("  ✗ Failed to open camera")
        return False

    print("  ✓ Camera opened successfully")
    print()

    # Test 3: Get camera properties
    print("[3/5] Querying camera properties...")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    backend = cap.getBackendName()

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  FOURCC: {fourcc}")
    print(f"  Backend: {backend}")
    print()

    # Test 4: Capture a frame
    print("[4/5] Capturing test frame...")
    ret, frame = cap.read()

    if not ret or frame is None:
        print("  ✗ Failed to capture frame")
        cap.release()
        return False

    print(f"  ✓ Frame captured successfully")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame dtype: {frame.dtype}")
    print()

    # Save test frame
    output_path = "test_frame.jpg"
    cv2.imwrite(output_path, frame)
    print(f"  ✓ Test frame saved to: {output_path}")
    print()

    # Release camera
    cap.release()

    # Test 5: Check Hailo device
    print("[5/5] Checking Hailo device availability...")
    try:
        from hailo_platform import VDevice

        params = VDevice.create_params()
        device = VDevice(params)

        print("  ✓ Hailo device created successfully")
        print(f"  Device ID: {device.get_id()}")

        # Get device info
        try:
            device_info = device.get_device_information()
            print(f"  Device architecture: {device_info.device_architecture}")
        except Exception:
            # Older API versions might not have get_device_information
            print("  Device info not available (older HailoRT version)")

        print()

    except ImportError as e:
        print(f"  ✗ Failed to import hailo_platform: {e}")
        print("  Make sure HailoRT is installed on your system")
        return False
    except Exception as e:
        print(f"  ✗ Failed to access Hailo device: {e}")
        print("  Check if the Hailo driver is loaded and you have permissions")
        return False

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("✓ All tests passed successfully!")
    print()
    print("Your system is ready to run the Hailo detector service.")
    print()
    return True


if __name__ == "__main__":
    device = "/dev/video0"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    success = test_camera(device)
    sys.exit(0 if success else 1)
