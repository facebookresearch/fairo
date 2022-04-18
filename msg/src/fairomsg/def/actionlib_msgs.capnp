@0xc7ead4e7826cfc1a;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::actionlib");
using Header = import "std_msgs.capnp".Header;
using Time = import "std_msgs.capnp".Time;
struct GoalID {
  stamp @0 :Time;
  id @1 :Text;
}
struct GoalStatus {
  goalId @0 :GoalID;
  status @1 :UInt8;
  const kPending :UInt8 = 0;
  const kActive :UInt8 = 1;
  const kPreempted :UInt8 = 2;
  const kSucceeded :UInt8 = 3;
  const kAborted :UInt8 = 4;
  const kRejected :UInt8 = 5;
  const kPreempting :UInt8 = 6;
  const kRecalling :UInt8 = 7;
  const kRecalled :UInt8 = 8;
  const kLost :UInt8 = 9;
  text @2 :Text;
}
struct GoalStatusArray {
  header @0 :Header;
  statusList @1 :List(GoalStatus);
}
