from geometry_msgs.msg import Point
import time

i = 0
while True:
    print(Point(x=i, y=i ** 2, z=i ** 3))
    i += 1
    time.sleep(1)
