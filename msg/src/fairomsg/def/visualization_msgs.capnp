@0xc6ab06e10596c625;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::visualization");
using ColorRGBA = import "std_msgs.capnp".ColorRGBA;
using Duration = import "std_msgs.capnp".Duration;
using Header = import "std_msgs.capnp".Header;
using Point = import "geometry_msgs.capnp".Point;
using Pose = import "geometry_msgs.capnp".Pose;
using Quaternion = import "geometry_msgs.capnp".Quaternion;
using Vector3 = import "geometry_msgs.capnp".Vector3;
struct InteractiveMarkerUpdate {
  serverId @0 :Text;
  seqNum @1 :UInt64;
  const kKeepAlive :UInt8 = 0;
  const kUpdate :UInt8 = 1;
  type @2 :UInt8;
  markers @3 :List(InteractiveMarker);
  poses @4 :List(InteractiveMarkerPose);
  erases @5 :List(Text);
}
struct ImageMarker {
  const kCircle :UInt8 = 0;
  const kLineStrip :UInt8 = 1;
  const kLineList :UInt8 = 2;
  const kPolygon :UInt8 = 3;
  const kPoints :UInt8 = 4;
  const kAdd :UInt8 = 0;
  const kRemove :UInt8 = 1;
  header @0 :Header;
  ns @1 :Text;
  id @2 :Int32;
  type @3 :Int32;
  action @4 :Int32;
  position @5 :Point;
  scale @6 :Float32;
  outlineColor @7 :ColorRGBA;
  filled @8 :UInt8;
  fillColor @9 :ColorRGBA;
  lifetime @10 :Duration;
  points @11 :List(Point);
  outlineColors @12 :List(ColorRGBA);
}
struct InteractiveMarkerControl {
  name @0 :Text;
  orientation @1 :Quaternion;
  const kInherit :UInt8 = 0;
  const kFixed :UInt8 = 1;
  const kViewFacing :UInt8 = 2;
  orientationMode @2 :UInt8;
  const kNone :UInt8 = 0;
  const kMenu :UInt8 = 1;
  const kButton :UInt8 = 2;
  const kMoveAxis :UInt8 = 3;
  const kMovePlane :UInt8 = 4;
  const kRotateAxis :UInt8 = 5;
  const kMoveRotate :UInt8 = 6;
  const kMove3D :UInt8 = 7;
  const kRotate3D :UInt8 = 8;
  const kMoveRotate3D :UInt8 = 9;
  interactionMode @3 :UInt8;
  alwaysVisible @4 :Bool;
  markers @5 :List(Marker);
  independentMarkerOrientation @6 :Bool;
  description @7 :Text;
}
struct InteractiveMarkerInit {
  serverId @0 :Text;
  seqNum @1 :UInt64;
  markers @2 :List(InteractiveMarker);
}
struct MarkerArray {
  markers @0 :List(Marker);
}
struct InteractiveMarkerFeedback {
  header @0 :Header;
  clientId @1 :Text;
  markerName @2 :Text;
  controlName @3 :Text;
  const kKeepAlive :UInt8 = 0;
  const kPoseUpdate :UInt8 = 1;
  const kMenuSelect :UInt8 = 2;
  const kButtonClick :UInt8 = 3;
  const kMouseDown :UInt8 = 4;
  const kMouseUp :UInt8 = 5;
  eventType @4 :UInt8;
  pose @5 :Pose;
  menuEntryId @6 :UInt32;
  mousePoint @7 :Point;
  mousePointValid @8 :Bool;
}
struct Marker {
  const kArrow :UInt8 = 0;
  const kCube :UInt8 = 1;
  const kSphere :UInt8 = 2;
  const kCylinder :UInt8 = 3;
  const kLineStrip :UInt8 = 4;
  const kLineList :UInt8 = 5;
  const kCubeList :UInt8 = 6;
  const kSphereList :UInt8 = 7;
  const kPoints :UInt8 = 8;
  const kTextViewFacing :UInt8 = 9;
  const kMeshResource :UInt8 = 10;
  const kTriangleList :UInt8 = 11;
  const kAdd :UInt8 = 0;
  const kModify :UInt8 = 0;
  const kDelete :UInt8 = 2;
  const kDeleteall :UInt8 = 3;
  header @0 :Header;
  ns @1 :Text;
  id @2 :Int32;
  type @3 :Int32;
  action @4 :Int32;
  pose @5 :Pose;
  scale @6 :Vector3;
  color @7 :ColorRGBA;
  lifetime @8 :Duration;
  frameLocked @9 :Bool;
  points @10 :List(Point);
  colors @11 :List(ColorRGBA);
  text @12 :Text;
  meshResource @13 :Text;
  meshUseEmbeddedMaterials @14 :Bool;
}
struct InteractiveMarker {
  header @0 :Header;
  pose @1 :Pose;
  name @2 :Text;
  description @3 :Text;
  scale @4 :Float32;
  menuEntries @5 :List(MenuEntry);
  controls @6 :List(InteractiveMarkerControl);
}
struct MenuEntry {
  id @0 :UInt32;
  parentId @1 :UInt32;
  title @2 :Text;
  command @3 :Text;
  const kFeedback :UInt8 = 0;
  const kRosrun :UInt8 = 1;
  const kRoslaunch :UInt8 = 2;
  commandType @4 :UInt8;
}
struct InteractiveMarkerPose {
  header @0 :Header;
  pose @1 :Pose;
  name @2 :Text;
}
