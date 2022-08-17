@0xaaab035ff1b75f91;
using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("mrp::sensor");
using Header = import "std_msgs.capnp".Header;
using Point32 = import "geometry_msgs.capnp".Point32;
using Quaternion = import "geometry_msgs.capnp".Quaternion;
using Time = import "std_msgs.capnp".Time;
using Transform = import "geometry_msgs.capnp".Transform;
using Twist = import "geometry_msgs.capnp".Twist;
using Vector3 = import "geometry_msgs.capnp".Vector3;
using Wrench = import "geometry_msgs.capnp".Wrench;
struct BatteryState {
  const kPowerSupplyStatusUnknown :UInt8 = 0;
  const kPowerSupplyStatusCharging :UInt8 = 1;
  const kPowerSupplyStatusDischarging :UInt8 = 2;
  const kPowerSupplyStatusNotCharging :UInt8 = 3;
  const kPowerSupplyStatusFull :UInt8 = 4;
  const kPowerSupplyHealthUnknown :UInt8 = 0;
  const kPowerSupplyHealthGood :UInt8 = 1;
  const kPowerSupplyHealthOverheat :UInt8 = 2;
  const kPowerSupplyHealthDead :UInt8 = 3;
  const kPowerSupplyHealthOvervoltage :UInt8 = 4;
  const kPowerSupplyHealthUnspecFailure :UInt8 = 5;
  const kPowerSupplyHealthCold :UInt8 = 6;
  const kPowerSupplyHealthWatchdogTimerExpire :UInt8 = 7;
  const kPowerSupplyHealthSafetyTimerExpire :UInt8 = 8;
  const kPowerSupplyTechnologyUnknown :UInt8 = 0;
  const kPowerSupplyTechnologyNimh :UInt8 = 1;
  const kPowerSupplyTechnologyLion :UInt8 = 2;
  const kPowerSupplyTechnologyLipo :UInt8 = 3;
  const kPowerSupplyTechnologyLife :UInt8 = 4;
  const kPowerSupplyTechnologyNicd :UInt8 = 5;
  const kPowerSupplyTechnologyLimn :UInt8 = 6;
  header @0 :Header;
  voltage @1 :Float32;
  temperature @2 :Float32;
  current @3 :Float32;
  charge @4 :Float32;
  capacity @5 :Float32;
  designCapacity @6 :Float32;
  percentage @7 :Float32;
  powerSupplyStatus @8 :UInt8;
  powerSupplyHealth @9 :UInt8;
  powerSupplyTechnology @10 :UInt8;
  present @11 :Bool;
  cellVoltage @12 :List(Float32);
  cellTemperature @13 :List(Float32);
  location @14 :Text;
  serialNumber @15 :Text;
}
struct CameraInfo {
  header @0 :Header;
  height @1 :UInt32;
  width @2 :UInt32;
  distortionModel @3 :Text;
  d @4 :List(Float64);
  k @5 :List(Float64);
  r @6 :List(Float64);
  p @7 :List(Float64);
  binningX @8 :UInt32;
  binningY @9 :UInt32;
  roi @10 :RegionOfInterest;
}
struct ChannelFloat32 {
  name @0 :Text;
  values @1 :List(Float32);
}
struct CompressedImage {
  header @0 :Header;
  format @1 :Text;
  data @2 :Data;
}
struct FluidPressure {
  header @0 :Header;
  fluidPressure @1 :Float64;
  variance @2 :Float64;
}
struct Illuminance {
  header @0 :Header;
  illuminance @1 :Float64;
  variance @2 :Float64;
}
struct Image {
  header @0 :Header;
  height @1 :UInt32;
  width @2 :UInt32;
  encoding @3 :Text;
  isBigendian @4 :UInt8;
  step @5 :UInt32;
  data @6 :Data;
}
struct Imu {
  header @0 :Header;
  orientation @1 :Quaternion;
  orientationCovariance @2 :List(Float64);
  angularVelocity @3 :Vector3;
  angularVelocityCovariance @4 :List(Float64);
  linearAcceleration @5 :Vector3;
  linearAccelerationCovariance @6 :List(Float64);
}
struct JointState {
  header @0 :Header;
  name @1 :List(Text);
  position @2 :List(Float64);
  velocity @3 :List(Float64);
  effort @4 :List(Float64);
}
struct Joy {
  header @0 :Header;
  axes @1 :List(Float32);
  buttons @2 :List(Int32);
}
struct JoyFeedback {
  const kTypeLed :UInt8 = 0;
  const kTypeRumble :UInt8 = 1;
  const kTypeBuzzer :UInt8 = 2;
  type @0 :UInt8;
  id @1 :UInt8;
  intensity @2 :Float32;
}
struct JoyFeedbackArray {
  array @0 :List(JoyFeedback);
}
struct LaserEcho {
  echoes @0 :List(Float32);
}
struct LaserScan {
  header @0 :Header;
  angleMin @1 :Float32;
  angleMax @2 :Float32;
  angleIncrement @3 :Float32;
  timeIncrement @4 :Float32;
  scanTime @5 :Float32;
  rangeMin @6 :Float32;
  rangeMax @7 :Float32;
  ranges @8 :List(Float32);
  intensities @9 :List(Float32);
}
struct MagneticField {
  header @0 :Header;
  magneticField @1 :Vector3;
  magneticFieldCovariance @2 :List(Float64);
}
struct MultiDOFJointState {
  header @0 :Header;
  jointNames @1 :List(Text);
  transforms @2 :List(Transform);
  twist @3 :List(Twist);
  wrench @4 :List(Wrench);
}
struct MultiEchoLaserScan {
  header @0 :Header;
  angleMin @1 :Float32;
  angleMax @2 :Float32;
  angleIncrement @3 :Float32;
  timeIncrement @4 :Float32;
  scanTime @5 :Float32;
  rangeMin @6 :Float32;
  rangeMax @7 :Float32;
  ranges @8 :List(LaserEcho);
  intensities @9 :List(LaserEcho);
}
struct NavSatFix {
  header @0 :Header;
  status @1 :NavSatStatus;
  latitude @2 :Float64;
  longitude @3 :Float64;
  altitude @4 :Float64;
  positionCovariance @5 :List(Float64);
  const kCovarianceTypeUnknown :UInt8 = 0;
  const kCovarianceTypeApproximated :UInt8 = 1;
  const kCovarianceTypeDiagonalKnown :UInt8 = 2;
  const kCovarianceTypeKnown :UInt8 = 3;
  positionCovarianceType @6 :UInt8;
}
struct NavSatStatus {
  const kStatusNoFix :Int8 = 1;
  const kStatusFix :Int8 = 0;
  const kStatusSbasFix :Int8 = 1;
  const kStatusGbasFix :Int8 = 2;
  status @0 :Int8;
  const kServiceGps :UInt16 = 1;
  const kServiceGlonass :UInt16 = 2;
  const kServiceCompass :UInt16 = 4;
  const kServiceGalileo :UInt16 = 8;
  service @1 :UInt16;
}
struct PointCloud {
  header @0 :Header;
  points @1 :List(Point32);
  channels @2 :List(ChannelFloat32);
}
struct PointCloud2 {
  header @0 :Header;
  height @1 :UInt32;
  width @2 :UInt32;
  fields @3 :List(PointField);
  isBigendian @4 :Bool;
  pointStep @5 :UInt32;
  rowStep @6 :UInt32;
  data @7 :Data;
  isDense @8 :Bool;
}
struct PointField {
  const kInt8 :UInt8 = 1;
  const kUint8 :UInt8 = 2;
  const kInt16 :UInt8 = 3;
  const kUint16 :UInt8 = 4;
  const kInt32 :UInt8 = 5;
  const kUint32 :UInt8 = 6;
  const kFloat32 :UInt8 = 7;
  const kFloat64 :UInt8 = 8;
  name @0 :Text;
  offset @1 :UInt32;
  datatype @2 :UInt8;
  count @3 :UInt32;
}
struct Range {
  header @0 :Header;
  const kUltrasound :UInt8 = 0;
  const kInfrared :UInt8 = 1;
  radiationType @1 :UInt8;
  fieldOfView @2 :Float32;
  minRange @3 :Float32;
  maxRange @4 :Float32;
  range @5 :Float32;
}
struct RegionOfInterest {
  xOffset @0 :UInt32;
  yOffset @1 :UInt32;
  height @2 :UInt32;
  width @3 :UInt32;
  doRectify @4 :Bool;
}
struct RelativeHumidity {
  header @0 :Header;
  relativeHumidity @1 :Float64;
  variance @2 :Float64;
}
struct Temperature {
  header @0 :Header;
  temperature @1 :Float64;
  variance @2 :Float64;
}
struct TimeReference {
  header @0 :Header;
  timeRef @1 :Time;
  source @2 :Text;
}
