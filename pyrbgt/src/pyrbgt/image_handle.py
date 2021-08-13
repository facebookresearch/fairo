from abc import ABC, abstractmethod


class Intrinsics(object):
    def __init__(self):
        self.fu = 0
        self.fv = 0
        self.ppu = 0
        self.ppv = 0
        self.width = 0
        self.height = 0


class ImageHandle(ABC):
    @abstractmethod
    def get_image(self):
        return NotImplemented

    @abstractmethod
    def get_intrinsics(self):
        return NotImplemented
