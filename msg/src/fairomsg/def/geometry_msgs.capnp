@0xd639476cdeb42f7a;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::geometry");
using Header = import "std_msgs.capnp".Header;
struct PoseStamped {
  header @0 :Header;
  pose @1 :Pose;
}
struct InertiaStamped {
  header @0 :Header;
  inertia @1 :Inertia;
}
struct Twist {
  linear @0 :Vector3;
  angular @1 :Vector3;
}
struct Vector3Stamped {
  header @0 :Header;
  vector @1 :Vector3;
}
struct Vector3 {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}
struct PoseWithCovarianceStamped {
  header @0 :Header;
  pose @1 :PoseWithCovariance;
}
struct Inertia {
  m @0 :Float64;
  com @1 :Vector3;
  ixx @2 :Float64;
  ixy @3 :Float64;
  ixz @4 :Float64;
  iyy @5 :Float64;
  iyz @6 :Float64;
  izz @7 :Float64;
}
struct Polygon {
  points @0 :List(Point32);
}
struct Pose2D {
  x @0 :Float64;
  y @1 :Float64;
  theta @2 :Float64;
}
struct Point32 {
  x @0 :Float32;
  y @1 :Float32;
  z @2 :Float32;
}
struct PolygonStamped {
  header @0 :Header;
  polygon @1 :Polygon;
}
struct QuaternionStamped {
  header @0 :Header;
  quaternion @1 :Quaternion;
}
struct AccelWithCovariance {
  accel @0 :Accel;
  covariance @1 :List(Float64);
}
struct TwistStamped {
  header @0 :Header;
  twist @1 :Twist;
}
struct Quaternion {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
  w @3 :Float64;
}
struct PoseWithCovariance {
  pose @0 :Pose;
  covariance @1 :List(Float64);
}
struct PointStamped {
  header @0 :Header;
  point @1 :Point;
}
struct TwistWithCovariance {
  twist @0 :Twist;
  covariance @1 :List(Float64);
}
struct Accel {
  linear @0 :Vector3;
  angular @1 :Vector3;
}
struct TwistWithCovarianceStamped {
  header @0 :Header;
  twist @1 :TwistWithCovariance;
}
struct Transform {
  translation @0 :Vector3;
  rotation @1 :Quaternion;
}
struct AccelStamped {
  header @0 :Header;
  accel @1 :Accel;
}
struct Wrench {
  force @0 :Vector3;
  torque @1 :Vector3;
}
struct AccelWithCovarianceStamped {
  header @0 :Header;
  accel @1 :AccelWithCovariance;
}
struct TransformStamped {
  header @0 :Header;
  childFrameId @1 :Text;
  transform @2 :Transform;
}
struct PoseArray {
  header @0 :Header;
  poses @1 :List(Pose);
}
struct Point {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}
struct WrenchStamped {
  header @0 :Header;
  wrench @1 :Wrench;
}
struct Pose {
  position @0 :Point;
  orientation @1 :Quaternion;
}
