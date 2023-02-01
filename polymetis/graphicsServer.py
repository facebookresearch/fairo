import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import time

# server = p.connect(p.SHARED_MEMORY_SERVER, 1, "")
# client = p.connect(p.SHARED_MEMORY, 3, "")

# server = p.connect(p.GUI_SERVER)
# client = p.connect(p.SHARED_MEMORY, 12348, "")

# print(f"========== Success: {client != -1}, ID: {client} ==========")

server = p.connect(p.GRAPHICS_SERVER)
client = p.connect(p.GRAPHICS_SERVER_TCP)
# for i in range(11346, 11348):
# i = 12348
# for i in range(12345, 99999):
#   client = p.connect(p.SHARED_MEMORY, i, "")
#   if client != -1:
#     print(f"========== Success! ID: {i} ==========")
#     break
#   else:
#     print("========== Failed ==========")
# print("========== Done. ==========")

import pdb

pdb.set_trace()
