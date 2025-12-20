.PHONY: all clean build install test download-model help

# Default target - build the .deb package
all: build

# Help target
help:
	@echo "Hailo Detector Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build the .deb package (default)"
	@echo "  make build    - Build the .deb package"
	@echo "  make install  - Install the package locally"
	@echo "  make test     - Run test scripts"
	@echo "  make download-model - Download YOLOv8n HEF model"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ src/*.egg-info
	rm -f ../*.deb ../*.changes ../*.buildinfo
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete"

# Build .deb package
build:
	@echo "Building Debian package..."
	@echo ""
	dpkg-buildpackage -us -uc -b
	@echo ""
	@echo "Build complete!"
	@echo "Package created: ../hailo-detector_1.0.0_arm64.deb"

# Install locally (for testing)
install:
	@if [ ! -f ../hailo-detector_*.deb ]; then \
		echo "Error: Package not found. Run 'make build' first."; \
		exit 1; \
	fi
	@echo "Installing hailo-detector package..."
	sudo dpkg -i ../hailo-detector_*.deb || sudo apt-get install -f -y
	@echo ""
	@echo "Installation complete!"
	@echo "Check status: systemctl status hailo-detector"

# Run tests
test:
	@echo "Running camera test..."
	@echo ""
	python3 scripts/test-camera.py
	@echo ""
	@echo "To test the full streaming pipeline, run:"
	@echo "  python3 scripts/test-stream.py"

# Download model manually
download-model:
	@echo "Downloading YOLOv8n HEF model..."
	@bash scripts/download-model.sh

# Uninstall the package
uninstall:
	@echo "Uninstalling hailo-detector..."
	sudo apt-get remove hailo-detector -y
	@echo "Uninstall complete"

# Purge (remove package and configuration)
purge:
	@echo "Purging hailo-detector (removes all data)..."
	sudo apt-get purge hailo-detector -y
	@echo "Purge complete"
