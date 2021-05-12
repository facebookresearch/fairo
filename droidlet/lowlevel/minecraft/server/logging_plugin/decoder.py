"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import struct


class Decoder:
    def __init__(self, fp):
        self.fp = fp
        self.count = 0

    def readByte(self):
        return self.readStructFmt(">b")

    def readUByte(self):
        return self.readStructFmt(">B")

    def readShort(self):
        return self.readStructFmt(">h")

    def readUShort(self):
        return self.readStructFmt(">H")

    def readInt(self):
        return self.readStructFmt(">i")

    def readUInt(self):
        return self.readStructFmt(">I")

    def readLong(self):
        return self.readStructFmt(">q")

    def readULong(self):
        return self.readStructFmt(">Q")

    def readFloat(self):
        return self.readStructFmt(">f")

    def readDouble(self):
        return self.readStructFmt(">d")

    def readRaw(self, n):
        buf = self.fp.read(n)
        assert n == len(buf)
        self.count += n
        return buf

    def readStructFmt(self, fmt):
        size = struct.calcsize(fmt)
        buf = self.fp.read(size)
        if len(buf) != size:
            raise EOFError
        self.count += size
        return struct.unpack(fmt, buf)[0]

    def readString(self):
        length = self.readShort()
        x = self.readRaw(length).decode("utf-8")
        assert self.readByte() == 0, "String not null-terminated: {}".format(x)
        return x

    def readIntPos(self):
        return (self.readLong(), self.readLong(), self.readLong())

    def readFloatPos(self):
        return (self.readDouble(), self.readDouble(), self.readDouble())

    def readLook(self):
        return (self.readFloat(), self.readFloat())

    def readItem(self):
        return (self.readShort(), self.readShort(), self.readShort())

    def readBlock(self):
        return (self.readByte(), self.readByte())
