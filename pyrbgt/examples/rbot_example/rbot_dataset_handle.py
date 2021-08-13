import numpy as np
import cv2
import os

from pyrbgt import ImageHandle, Intrinsics


class RBOTDatasetHandle(ImageHandle):
    def __init__(self, dataset_path, object_name, sequence_name):
        self.dataset_path = dataset_path
        self.object_name = object_name
        self.sequence_name = sequence_name
        self.load_index = 0
        self.max_load_index = 1000

        intrinsic_path = os.path.join(dataset_path, "camera_calibration.txt")
        try:
            f = open(intrinsic_path, "r")
        except OSError:
            print("Cannot Open File at {}!".format(intrinsic_path))

        with f:
            intrinsic_lst = f.read().split("\n")[1].split("\t")
            self.intrinsics = Intrinsics()
            self.intrinsics.fu = float(intrinsic_lst[0])
            self.intrinsics.fv = float(intrinsic_lst[1])
            self.intrinsics.ppu = float(intrinsic_lst[2])
            self.intrinsics.ppv = float(intrinsic_lst[3])
            self.intrinsics.width = 640
            self.intrinsics.height = 512
            f.close()

    def get_intrinsics(self):
        return self.intrinsics

    def get_image(self):
        if self.load_index > self.max_load_index:
            return None
        image_path = os.path.join(
            self.dataset_path,
            self.object_name,
            "frames",
            "{}{}.png".format(self.sequence_name, str(int(self.load_index)).zfill(4)),
        )
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.load_index += 1
        return image
