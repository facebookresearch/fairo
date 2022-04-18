@0xe52fc95e987ddfdd;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::trajectory");
using Duration = import "std_msgs.capnp".Duration;
using Header = import "std_msgs.capnp".Header;
using Transform = import "geometry_msgs.capnp".Transform;
using Twist = import "geometry_msgs.capnp".Twist;
struct JointTrajectory {
  header @0 :Header;
  jointNames @1 :List(Text);
  points @2 :List(JointTrajectoryPoint);
}
struct MultiDOFJointTrajectoryPoint {
  transforms @0 :List(Transform);
  velocities @1 :List(Twist);
  accelerations @2 :List(Twist);
  timeFromStart @3 :Duration;
}
struct MultiDOFJointTrajectory {
  header @0 :Header;
  jointNames @1 :List(Text);
  points @2 :List(MultiDOFJointTrajectoryPoint);
}
struct JointTrajectoryPoint {
  positions @0 :List(Float64);
  velocities @1 :List(Float64);
  accelerations @2 :List(Float64);
  effort @3 :List(Float64);
  timeFromStart @4 :Duration;
}
