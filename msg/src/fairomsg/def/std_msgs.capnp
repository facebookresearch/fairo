@0xfcdd69141cab2608;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::std");
struct Int16MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(Int16);
}
struct Time {
  sec @0 :UInt32;
  nsec @1 :UInt32;
}
struct MultiArrayDimension {
  label @0 :Text;
  size @1 :UInt32;
  stride @2 :UInt32;
}
struct Float32MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(Float32);
}
struct UInt32MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(UInt32);
}
struct ColorRGBA {
  r @0 :Float32;
  g @1 :Float32;
  b @2 :Float32;
  a @3 :Float32;
}
struct Int32MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(Int32);
}
struct String {
  data @0 :Text;
}
struct UInt16MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(UInt16);
}
struct UInt8MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :Data;
}
struct Byte {
  data @0 :UInt8;
}
struct Header {
  seq @0 :UInt32;
  stamp @1 :Time;
  frameId @2 :Text;
}
struct Float64MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(Float64);
}
struct Char {
  data @0 :Int8;
}
struct UInt64MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(UInt64);
}
struct Duration {
  sec @0 :Int32;
  nsec @1 :Int32;
}
struct MultiArrayLayout {
  dim @0 :List(MultiArrayDimension);
  dataOffset @1 :UInt32;
}
struct Int8MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :Data;
}
struct Int64MultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :List(Int64);
}
struct ByteMultiArray {
  layout @0 :MultiArrayLayout;
  data @1 :Data;
}
struct Empty {
}
