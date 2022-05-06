@0x8e1824e69341e9dd;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::nav");
using GoalID = import "actionlib_msgs.capnp".GoalID;
using GoalStatus = import "actionlib_msgs.capnp".GoalStatus;
using Header = import "std_msgs.capnp".Header;
using Point = import "geometry_msgs.capnp".Point;
using Pose = import "geometry_msgs.capnp".Pose;
using PoseStamped = import "geometry_msgs.capnp".PoseStamped;
using PoseWithCovariance = import "geometry_msgs.capnp".PoseWithCovariance;
using Time = import "std_msgs.capnp".Time;
using TwistWithCovariance = import "geometry_msgs.capnp".TwistWithCovariance;
struct Path {
  header @0 :Header;
  poses @1 :List(PoseStamped);
}
struct OccupancyGrid {
  header @0 :Header;
  info @1 :MapMetaData;
  data @2 :Data;
}
struct GetMapGoal {
}
struct GetMapActionResult {
  header @0 :Header;
  status @1 :GoalStatus;
  result @2 :GetMapResult;
}
struct GetMapFeedback {
}
struct GridCells {
  header @0 :Header;
  cellWidth @1 :Float32;
  cellHeight @2 :Float32;
  cells @3 :List(Point);
}
struct GetMapResult {
  map @0 :OccupancyGrid;
}
struct MapMetaData {
  mapLoadTime @0 :Time;
  resolution @1 :Float32;
  width @2 :UInt32;
  height @3 :UInt32;
  origin @4 :Pose;
}
struct Odometry {
  header @0 :Header;
  childFrameId @1 :Text;
  pose @2 :PoseWithCovariance;
  twist @3 :TwistWithCovariance;
}
struct GetMapActionGoal {
  header @0 :Header;
  goalId @1 :GoalID;
  goal @2 :GetMapGoal;
}
struct GetMapActionFeedback {
  header @0 :Header;
  status @1 :GoalStatus;
  feedback @2 :GetMapFeedback;
}
struct GetMapAction {
  actionGoal @0 :GetMapActionGoal;
  actionResult @1 :GetMapActionResult;
  actionFeedback @2 :GetMapActionFeedback;
}
