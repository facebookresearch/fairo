import geometry_msgs.msg
import rospy
import tf2_ros


class TransformTree:
    def __init__(self):
        self._buffer = tf2_ros.BufferCore(rospy.Duration(10.0))

    def set_transform(self, frame_a, frame_b, timestamp, tf_dict):
        tfs = geometry_msgs.msg.TransformStamped()
        tfs.header.stamp = timestamp
        tfs.header.frame_id = frame_a
        tfs.child_frame_id = frame_b
        tfs.transform.translation.x = tf_dict.get("translation", {}).get("x", 0)
        tfs.transform.translation.y = tf_dict.get("translation", {}).get("y", 0)
        tfs.transform.translation.z = tf_dict.get("translation", {}).get("z", 0)
        tfs.transform.rotation.x = tf_dict.get("rotation", {}).get("x", 0)
        tfs.transform.rotation.y = tf_dict.get("rotation", {}).get("y", 0)
        tfs.transform.rotation.z = tf_dict.get("rotation", {}).get("z", 0)
        tfs.transform.rotation.w = tf_dict.get("rotation", {}).get("w", 1)
        self._buffer.set_transform(tfs, "")

    def lookup_transform(self, frame_a, frame_b, timestamp):
        return self._buffer.lookup_transform_core(frame_a, frame_b, timestamp)


tree = TransformTree()
tree.set_transform("a", "b", rospy.Time(0), {"translation": {"x": 1}})
tree.set_transform("b", "c", rospy.Time(0), {"translation": {"x": 1}})
print(tree.lookup_transform("a", "c", rospy.Time(0)))
