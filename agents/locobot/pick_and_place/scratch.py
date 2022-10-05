import numpy as np


depth = np.load("depth.npy")
rgb = np.load("rgb.npy")

print(depth.shape, depth.min(), depth.max())
print(rgb.shape, rgb.min(), rgb.max())
