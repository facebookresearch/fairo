import argparse
import torch, cv2, os
import numpy as np
from PIL import Image
from torchvision import transforms


class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self._n = 0
        self._value = 0
    
    def add(self, value):
        self._value += value
        self._n += 1
    
    @property
    def mean(self):
        if self._n:
            return self._value / self._n
        return 0


class Arguments:
    def __init__(self, output_folder, **kwargs):
        self.output_folder = output_folder
        args = self.default
        for k in kwargs.keys():
            if k not in args:
                print("WARNING: key {} not in defaults!".format(k))

        args.update(kwargs)
        for k, v in args.items():
            setattr(self, k, v)
    
    def default(self):
        raise NotImplementedError


def args2cmd(Args):
    defaults = Args(None).default
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder')
    for k, v in defaults.items():
        if isinstance(v, bool):
            assert v == False, "default for store_true is false"
            parser.add_argument('--' + k, action='store_true')
        else:
            parser.add_argument('--' + k, default=v, type=type(v), help='default {}'.format(v))
    
    args = parser.parse_args()
    if os.path.exists(args.output_folder):
        input('Path already exists! Press enter to continue')
    return args


def build_transform(size=(120, 160)):
    transform = transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])
    def f(image):
        image = Image.fromarray(image)
        return transform(image)
    return f
