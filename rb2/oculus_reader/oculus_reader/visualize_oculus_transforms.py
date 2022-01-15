from reader import OculusReader
from tf.transformations import quaternion_from_matrix
import rospy
import tf2_ros
import geometry_msgs.msg


def publish_transform(transform, name):
    translation = transform[:3, 3]

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'world'
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = quaternion_from_matrix(transform)
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    br.sendTransform(t)


def main():
    oculus_reader = OculusReader()
    rospy.init_node('oculus_reader')

    while not rospy.is_shutdown():
        rospy.sleep(1)
        transformations, buttons = oculus_reader.get_transformations_and_buttons()
        if 'r' not in transformations:
            continue

        right_controller_pose = transformations['r']
        publish_transform(right_controller_pose, 'oculus')
        print('transformations', transformations)
        print('buttons', buttons)

if __name__ == '__main__':
    main()
