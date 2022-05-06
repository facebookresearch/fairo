@0xf072c9baec2e3f22;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::diagnostic");
using Header = import "std_msgs.capnp".Header;
struct DiagnosticStatus {
  const kOk :UInt8 = 0;
  const kWarn :UInt8 = 1;
  const kError :UInt8 = 2;
  const kStale :UInt8 = 3;
  level @0 :UInt8;
  name @1 :Text;
  message @2 :Text;
  hardwareId @3 :Text;
  values @4 :List(KeyValue);
}
struct DiagnosticArray {
  header @0 :Header;
  status @1 :List(DiagnosticStatus);
}
struct KeyValue {
  key @0 :Text;
  value @1 :Text;
}
