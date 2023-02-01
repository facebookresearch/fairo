import pybullet as p

p.connect(p.SHARED_MEMORY, hostName="localhost", port=6007)
print("here")
