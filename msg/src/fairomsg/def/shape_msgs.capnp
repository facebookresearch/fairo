@0xd95d03f2b0af9bb0;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::shape");
using Point = import "geometry_msgs.capnp".Point;
struct SolidPrimitive {
  const kBox :UInt8 = 1;
  const kSphere :UInt8 = 2;
  const kCylinder :UInt8 = 3;
  const kCone :UInt8 = 4;
  type @0 :UInt8;
  dimensions @1 :List(Float64);
  const kBoxX :UInt8 = 0;
  const kBoxY :UInt8 = 1;
  const kBoxZ :UInt8 = 2;
  const kSphereRadius :UInt8 = 0;
  const kCylinderHeight :UInt8 = 0;
  const kCylinderRadius :UInt8 = 1;
  const kConeHeight :UInt8 = 0;
  const kConeRadius :UInt8 = 1;
}
struct Plane {
  coef @0 :List(Float64);
}
struct MeshTriangle {
  vertexIndices @0 :List(UInt32);
}
struct Mesh {
  triangles @0 :List(MeshTriangle);
  vertices @1 :List(Point);
}
