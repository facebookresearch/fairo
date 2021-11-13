from geometry_msgs.msg import Point
from my_msgs.py import PsUtil
import time

i = 0
while True:
    print(Point(x=i, y=i ** 2, z=i ** 3))
    print(PsUtil(cpu_usage=i, mem_usage=i ** 0.5))
    i += 1
    time.sleep(1)
