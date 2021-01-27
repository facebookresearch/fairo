import os
import sys

dir_test = os.path.dirname(__file__)
dir_base_agent = os.path.join(dir_test, "..")
dir_root = os.path.join(dir_base_agent, "..")


sys.path.append(dir_root)
sys.path.insert(0, dir_test)
sys.path.insert(0, dir_base_agent)

print("sys path {}".format(sys.path))
