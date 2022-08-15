import torch
import clip
from PIL import Image

import time

time0 = time.time()
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["where is the cube"] * 100).to(device)
text_features = model.encode_text(text)

print(text_features.size())
print(time.time() - time0)
