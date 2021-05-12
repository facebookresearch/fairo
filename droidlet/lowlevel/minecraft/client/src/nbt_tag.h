// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <stdint.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define TAG_End 0
#define TAG_Byte 1
#define TAG_Short 2
#define TAG_Int 3
#define TAG_Long 4
#define TAG_Float 5
#define TAG_Double 6
#define TAG_Byte_Array 7
#define TAG_String 8
#define TAG_List 9
#define TAG_Compound 10
#define TAG_Int_Array 11
#define TAG_Long_Array 12

// This class implements a parser for the NBT file format, outlined here:
//
// https://minecraft.gamepedia.com/NBT_format
//
// This class is currently INCOMPLETE.
//
class NBTTag {
 public:
  NBTTag() {}

  // Parse raw data from *p and return a complete NBTTag.
  // N.B. after this call, *p will point to the first byte after the complete NBT tag.
  static std::shared_ptr<NBTTag> from(uint8_t** p);

  // Return the tag id (type) of the tag
  uint8_t getId() { return id_; }

  // Return the name of the tag (or empty string if it has no name)
  std::string getName() { return name_; }

 protected:
  static std::string readName(uint8_t** p);
  NBTTag(uint8_t id) : id_(id) {}
  NBTTag(uint8_t id, std::string name) : id_(id), name_(name) {}

  uint8_t id_;
  std::string name_;
};

class NBTTagEnd : public NBTTag {
 public:
  NBTTagEnd() : NBTTag(TAG_End) {}
};

class NBTTagByte : public NBTTag {
 public:
  static int8_t readVal(uint8_t** p);
  NBTTagByte(uint8_t** p);
  NBTTagByte(int8_t v) : NBTTag(TAG_Byte), val_(v) {}
  int8_t getVal() { return val_; }

 private:
  int8_t val_;
};

class NBTTagInt : public NBTTag {
 public:
  static int32_t readVal(uint8_t** p);
  NBTTagInt(uint8_t** p);
  NBTTagInt(int32_t v) : NBTTag(TAG_Int), val_(v) {}
  int32_t getVal() { return val_; }

 private:
  int32_t val_;
};

class NBTTagFloat : public NBTTag {
 public:
  static float readVal(uint8_t** p);
  NBTTagFloat(uint8_t** p);
  NBTTagFloat(float v) : NBTTag(TAG_Float), val_(v) {}
  float getVal() { return val_; }

 private:
  float val_;
};

class NBTTagDouble : public NBTTag {
 public:
  static double readVal(uint8_t** p);
  NBTTagDouble(uint8_t** p);
  NBTTagDouble(double v) : NBTTag(TAG_Double), val_(v) {}
  double getVal() { return val_; }

 private:
  double val_;
};

class NBTTagShort : public NBTTag {
 public:
  static int16_t readVal(uint8_t** p);
  NBTTagShort(uint8_t** p);
  NBTTagShort(int16_t v) : NBTTag(TAG_Short), val_(v) {}
  int16_t getVal() { return val_; }

 private:
  int16_t val_;
};

class NBTTagLong : public NBTTag {
 public:
  static int64_t readVal(uint8_t** p);
  NBTTagLong(uint8_t** p);
  NBTTagLong(int64_t v) : NBTTag(TAG_Long), val_(v) {}
  int64_t getVal() { return val_; }

 private:
  int64_t val_;
};

class NBTTagByteArray : public NBTTag {
 public:
  static std::vector<uint8_t> readVal(uint8_t** p);
  NBTTagByteArray(uint8_t** p);
  NBTTagByteArray(std::vector<uint8_t> v) : NBTTag(TAG_Byte_Array), val_(v) {}
  std::vector<uint8_t> getVal() { return val_; }

 private:
  std::vector<uint8_t> val_;
};

class NBTTagString : public NBTTag {
 public:
  static std::string readVal(uint8_t** p);
  NBTTagString(uint8_t** p);
  NBTTagString(std::string v) : NBTTag(TAG_String), val_(v) {}
  std::string getVal() { return val_; }

 private:
  std::string val_;
};

class NBTTagList : public NBTTag {
 public:
  NBTTagList(uint8_t** p);
  uint8_t valsType() { return valsType_; }
  unsigned long size() { return vals_.size(); }
  const std::vector<std::shared_ptr<NBTTag>>& getVals() const { return vals_; }

 private:
  uint8_t valsType_;
  std::vector<std::shared_ptr<NBTTag>> vals_;
};

class NBTTagCompound : public NBTTag {
 public:
  static std::map<std::string, std::shared_ptr<NBTTag>> readVal(uint8_t** p);
  NBTTagCompound(uint8_t** p);
  NBTTagCompound(std::map<std::string, std::shared_ptr<NBTTag>> v)
      : NBTTag(TAG_Compound), vals_(v) {}
  const std::map<std::string, std::shared_ptr<NBTTag>> getVals() const { return vals_; }

  // Return child by name, or NULL if there is no child by that name
  std::shared_ptr<NBTTag> getChild(const std::string& name);

 private:
  std::map<std::string, std::shared_ptr<NBTTag>> vals_;
};

class NBTTagIntArray : public NBTTag {
 public:
  static std::vector<int32_t> readVal(uint8_t** p);
  NBTTagIntArray(uint8_t** p);
  NBTTagIntArray(std::vector<int32_t> v) : NBTTag(TAG_Int_Array), val_(v) {}
  std::vector<int32_t> getVal() { return val_; }

 private:
  std::vector<int32_t> val_;
};
