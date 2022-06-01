from rplidar import RPLidar
import sys
import threading
import time
import traceback

if __name__ == "__main__":
    from lidar_abc import LidarABC
else:
    from .lidar_abc import LidarABC


class Lidar(LidarABC):
    def __init__(self):
        self._lidar = RPLidar("/dev/hello-lrf")
        self._lock = threading.Lock()
        self._thread = None
        self.latest_scan = None

    def start(self):
        self._thread = threading.Thread(target=self.lidar_loop, daemon=True)
        self._thread.start()

    def get_latest_scan(self):
        with self._lock:
            return self.latest_scan

    def lidar_loop(self):
        while True:
            try:
                for scan in self._lidar.iter_scans():
                    timestamp = time.time()
                    with self._lock:
                        self.latest_scan = (timestamp, scan)
            except:
                traceback.print_exc()
                self._lidar.stop()
                self._lidar.stop_motor()
                # just keep going
