@0xffe8a4b298f1916c;
using Cxx = import "/capnp/c++.capnp";
struct CollisionRequest {
  pcd @0 :Data;
  grasps @1 :Data;
}