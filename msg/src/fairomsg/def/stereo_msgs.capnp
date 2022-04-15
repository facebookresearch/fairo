@0xa9f896a24e23368e;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::stereo");
using Header = import "std_msgs.capnp".Header;
using Image = import "sensor_msgs.capnp".Image;
using RegionOfInterest = import "sensor_msgs.capnp".RegionOfInterest;
struct DisparityImage {
  header @0 :Header;
  image @1 :Image;
  f @2 :Float32;
  t @3 :Float32;
  validWindow @4 :RegionOfInterest;
  minDisparity @5 :Float32;
  maxDisparity @6 :Float32;
  deltaD @7 :Float32;
}
