import math
import sys
import threading
import time
import traceback
import rospy
from sensor_msgs.msg import LaserScan


class Lidar:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self.latest_scan = None

    def start(self):
        try:
            rospy.init_node("droidlet_lidar_node")
        except:
            pass
        self._thread = threading.Thread(target=self.lidar_loop, daemon=True)
        self._thread.start()

    def get_latest_scan(self):
        with self._lock:
            return self.latest_scan

    def _scan_callback(self, scan):
        with self._lock:
            in_mm = [s * 1000 for s in scan.ranges]
            angle_min = scan.angle_min
            angle_max = scan.angle_max
            angle_inc = scan.angle_increment
            angles = []
            for i in range(len(in_mm)):
                radians = angle_min + angle_inc * i
                angles.append(math.degrees(radians) + 180)
            quality = scan.intensities
            lscan = []
            for i in range(len(in_mm)):
                lscan.append((quality[i], angles[i], in_mm[i]))
            self.latest_scan = (scan.header.stamp.secs, lscan)

    def lidar_loop(self):
        rospy.Subscriber("scan_filtered", LaserScan, self._scan_callback, queue_size=1)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("droidlet_lidar_node")
    Lidar()
    while not rospy.is_shutdown():
        time.sleep(10)
