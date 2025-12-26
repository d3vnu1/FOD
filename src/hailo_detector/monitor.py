"""
System monitoring module for FleeKey Object Detection.
Provides CPU usage, CPU temperature, Hailo device temperature, and memory stats.
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SystemMonitor:
    """System resource and temperature monitor."""

    def __init__(self):
        self._hailo_available = None
        self._hailo_device = None
        self._last_cpu_times: Optional[Tuple[float, float]] = None
        self._last_cpu_check = 0
        self._cpu_usage = 0.0

    def _init_hailo(self):
        """Initialize Hailo device for temperature monitoring."""
        if self._hailo_available is False:
            return False

        try:
            from hailo_platform import Device
            devices = Device.scan()
            if devices:
                self._hailo_device = Device(devices[0])
                self._hailo_available = True
                logger.info("Hailo temperature monitoring initialized")
                return True
            else:
                self._hailo_available = False
                return False
        except Exception as e:
            logger.warning(f"Failed to init Hailo monitoring: {e}")
            self._hailo_available = False
            return False

    def get_cpu_usage(self) -> float:
        """
        Get CPU usage percentage.

        Returns:
            CPU usage as percentage (0-100)
        """
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()

            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq, steal
            user = float(parts[1])
            nice = float(parts[2])
            system = float(parts[3])
            idle = float(parts[4])
            iowait = float(parts[5]) if len(parts) > 5 else 0
            irq = float(parts[6]) if len(parts) > 6 else 0
            softirq = float(parts[7]) if len(parts) > 7 else 0
            steal = float(parts[8]) if len(parts) > 8 else 0

            total = user + nice + system + idle + iowait + irq + softirq + steal
            busy = total - idle - iowait

            if self._last_cpu_times is not None:
                last_total, last_busy = self._last_cpu_times
                total_diff = total - last_total
                busy_diff = busy - last_busy

                if total_diff > 0:
                    self._cpu_usage = (busy_diff / total_diff) * 100.0

            self._last_cpu_times = (total, busy)
            return self._cpu_usage

        except Exception as e:
            logger.warning(f"Failed to read CPU usage: {e}")
            return 0.0

    def get_cpu_load(self) -> Dict[str, float]:
        """
        Get CPU load averages.

        Returns:
            Dict with load_1m, load_5m, load_15m
        """
        try:
            with open('/proc/loadavg', 'r') as f:
                parts = f.read().strip().split()
                return {
                    'load_1m': float(parts[0]),
                    'load_5m': float(parts[1]),
                    'load_15m': float(parts[2])
                }
        except Exception as e:
            logger.warning(f"Failed to read CPU load: {e}")
            return {'load_1m': 0.0, 'load_5m': 0.0, 'load_15m': 0.0}

    def get_cpu_temp(self) -> Optional[float]:
        """
        Get CPU temperature in Celsius.

        Returns:
            Temperature in Celsius or None if unavailable
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = int(f.read().strip())
                return temp_millidegrees / 1000.0
        except Exception as e:
            logger.warning(f"Failed to read CPU temperature: {e}")
            return None

    def get_hailo_temp(self) -> Optional[float]:
        """
        Get Hailo device temperature in Celsius.

        Returns:
            Temperature in Celsius or None if unavailable
        """
        if self._hailo_available is None:
            self._init_hailo()

        if not self._hailo_available or self._hailo_device is None:
            return None

        try:
            temp_info = self._hailo_device.control.get_chip_temperature()
            # Average of both temperature sensors
            avg_temp = (temp_info.ts0_temperature + temp_info.ts1_temperature) / 2.0
            return avg_temp
        except Exception as e:
            logger.warning(f"Failed to read Hailo temperature: {e}")
            # Try to reinitialize on next call
            self._hailo_available = None
            self._hailo_device = None
            return None

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage in MB.

        Returns:
            Dict with total, used, available memory in MB
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1])
                        meminfo[key] = value

                total = meminfo.get('MemTotal', 0) // 1024
                available = meminfo.get('MemAvailable', 0) // 1024
                used = total - available

                return {
                    'total_mb': total,
                    'used_mb': used,
                    'available_mb': available
                }
        except Exception as e:
            logger.warning(f"Failed to read memory info: {e}")
            return {'total_mb': 0, 'used_mb': 0, 'available_mb': 0}

    def get_all_stats(self) -> Dict:
        """
        Get all system statistics.

        Returns:
            Dict with all monitoring data
        """
        cpu_load = self.get_cpu_load()
        cpu_temp = self.get_cpu_temp()
        cpu_usage = self.get_cpu_usage()
        hailo_temp = self.get_hailo_temp()
        memory = self.get_memory_usage()

        return {
            'cpu': {
                'usage': round(cpu_usage, 1),
                'load_1m': round(cpu_load['load_1m'], 2),
                'load_5m': round(cpu_load['load_5m'], 2),
                'load_15m': round(cpu_load['load_15m'], 2),
                'temperature': round(cpu_temp, 1) if cpu_temp else None
            },
            'hailo': {
                'temperature': round(hailo_temp, 1) if hailo_temp else None
            },
            'memory': memory
        }

    def cleanup(self):
        """Release resources."""
        if self._hailo_device is not None:
            try:
                self._hailo_device.release()
            except:
                pass
            self._hailo_device = None
