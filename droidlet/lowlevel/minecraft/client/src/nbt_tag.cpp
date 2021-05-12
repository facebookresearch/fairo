// Copyright (c) Facebook, Inc. and its affiliates.

#include <glog/logging.h>

#include "../../client/src/big_endian.h"
#include "nbt_tag.h"

using namespace std;

// NBTTag

shared_ptr<NBTTag> NBTTag::from(uint8_t** p) {
  int8_t id = BigEndian::readIntType<int8_t>(*p);
  *p += 1;

  // TODO: the rest of the tags
  switch (id) {
    case TAG_End:
      return make_shared<NBTTagEnd>();
    case TAG_Byte:
      return make_shared<NBTTagByte>(p);
    case TAG_Short:
      return make_shared<NBTTagShort>(p);
    case TAG_Int:
      return make_shared<NBTTagInt>(p);
    case TAG_Long:
      return make_shared<NBTTagLong>(p);
    case TAG_Float:
      return make_shared<NBTTagFloat>(p);
    case TAG_Double:
      return make_shared<NBTTagDouble>(p);
    case TAG_Byte_Array:
      return make_shared<NBTTagByteArray>(p);
    case TAG_String:
      return make_shared<NBTTagString>(p);
    case TAG_List:
      return make_shared<NBTTagList>(p);
    case TAG_Compound:
      return make_shared<NBTTagCompound>(p);
    case TAG_Int_Array:
      return make_shared<NBTTagIntArray>(p);
    default:
      LOG(FATAL) << "Not implemented: tag id " << (int)id;
  }
}

string NBTTag::readName(uint8_t** p) {
  auto nameLen = BigEndian::readIntType<int16_t>(*p);
  *p += 2;
  string s(*p, *p + nameLen);
  *p += nameLen;
  return s;
}

// NBTTagByte

int8_t NBTTagByte::readVal(uint8_t** p) {
  int8_t b = BigEndian::readIntType<int8_t>(*p);
  *p += 1;
  return b;
}

NBTTagByte::NBTTagByte(uint8_t** p) : NBTTag(TAG_Byte) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagShort

int16_t NBTTagShort::readVal(uint8_t** p) {
  int16_t b = BigEndian::readIntType<int16_t>(*p);
  *p += 2;
  return b;
}

NBTTagShort::NBTTagShort(uint8_t** p) : NBTTag(TAG_Short) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagInt

int32_t NBTTagInt::readVal(uint8_t** p) {
  int32_t b = BigEndian::readIntType<int32_t>(*p);
  *p += 4;
  return b;
}

NBTTagInt::NBTTagInt(uint8_t** p) : NBTTag(TAG_Int) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagLong

int64_t NBTTagLong::readVal(uint8_t** p) {
  int64_t b = BigEndian::readIntType<int64_t>(*p);
  *p += 8;
  return b;
}

NBTTagLong::NBTTagLong(uint8_t** p) : NBTTag(TAG_Long) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagFloat

float NBTTagFloat::readVal(uint8_t** p) {
  float b = BigEndian::readFloat(*p);
  *p += 4;
  return b;
}

NBTTagFloat::NBTTagFloat(uint8_t** p) : NBTTag(TAG_Float) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagDouble

double NBTTagDouble::readVal(uint8_t** p) {
  double b = BigEndian::readDouble(*p);
  *p += 8;
  return b;
}

NBTTagDouble::NBTTagDouble(uint8_t** p) : NBTTag(TAG_Double) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagByteArray

vector<uint8_t> NBTTagByteArray::readVal(uint8_t** p) {
  int size = NBTTagInt::readVal(p);
  vector<uint8_t> v(*p, *p + size);
  *p += size;
  return v;
}

NBTTagByteArray::NBTTagByteArray(uint8_t** p) : NBTTag(TAG_Byte_Array) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagString

string NBTTagString::readVal(uint8_t** p) { return readName(p); }

NBTTagString::NBTTagString(uint8_t** p) : NBTTag(TAG_String) {
  name_ = readName(p);
  val_ = readVal(p);
}

// NBTTagList

NBTTagList::NBTTagList(uint8_t** p) : NBTTag(TAG_List) {
  name_ = readName(p);
  valsType_ = BigEndian::readIntType<int8_t>(*p);
  *p += 1;
  int valSize = BigEndian::readIntType<int32_t>(*p);
  *p += 4;
  for (int i = 0; i < valSize; i++) {
    switch (valsType_) {
      case TAG_Byte: {
        int8_t v;
        v = NBTTagByte::readVal(p);
        vals_.push_back(make_shared<NBTTagByte>(v));
        break;
      }
      case TAG_Float: {
        float v;
        v = NBTTagFloat::readVal(p);
        vals_.push_back(make_shared<NBTTagFloat>(v));
        break;
      }
      case TAG_Double: {
        double v;
        v = NBTTagDouble::readVal(p);
        vals_.push_back(make_shared<NBTTagDouble>(v));
        break;
      }
      case TAG_Compound: {
        map<string, shared_ptr<NBTTag>> m;
        m = NBTTagCompound::readVal(p);
        vals_.push_back(make_shared<NBTTagCompound>(m));
        break;
      }
      default:
        LOG(FATAL) << "Not implemented: list of tag id " << (int)valsType_;
    }
  }
}

// NBTTagCompound

map<string, shared_ptr<NBTTag>> NBTTagCompound::readVal(uint8_t** p) {
  map<string, shared_ptr<NBTTag>> m;
  while (true) {
    shared_ptr<NBTTag> tag = NBTTag::from(p);
    if (tag->getId() == TAG_End) {
      return m;
    }
    m[tag->getName()] = tag;
  }
}

NBTTagCompound::NBTTagCompound(uint8_t** p) : NBTTag(TAG_Compound) {
  name_ = readName(p);
  vals_ = readVal(p);
}

shared_ptr<NBTTag> NBTTagCompound::getChild(const string& name) {
  auto f = vals_.find(name);
  if (f == vals_.end()) {
    return NULL;
  } else {
    return f->second;
  }
}

// NBTTagIntArray

vector<int32_t> NBTTagIntArray::readVal(uint8_t** p) {
  int size = NBTTagInt::readVal(p);
  vector<int32_t> v(size);
  for (int i = 0; i < size; i++, *p += 4) {
    v[i] = BigEndian::readIntType<int32_t>(*p);
  }
  return v;
}

NBTTagIntArray::NBTTagIntArray(uint8_t** p) : NBTTag(TAG_Int_Array) {
  name_ = readName(p);
  val_ = readVal(p);
}
