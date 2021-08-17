# for using the locobot bumper sensor data
from kobuki_msgs.msg import BumperEvent
import rospy


class BumperCallbacks(object):
    def __init__(self):
        self.bumper_state = set()
        rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.bumper_callback)

    def bumper_callback(self, msg):
        if msg.state == 0:
            self.bumper_state.discard(msg.bumper)
        elif msg.state == 1:
            self.bumper_state.add(msg.bumper)


if __name__ == "__main__":
    rospy.init_node("test")
    bumper = BumperCallbacks()
