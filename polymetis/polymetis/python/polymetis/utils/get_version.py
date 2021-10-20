import os
import json

version = ""

# Conda installed
stream = os.popen("conda list | grep polymetis")
polymetis_info = ""
for line in stream:
    if "polymetis" in line:
        polymetis_info = line.strip("\n")
polymetis_info_ls = [s for s in polymetis_info.split(" ") if len(s) > 0]
if polymetis_info_ls[-1] == "fair-robotics":
    version = polymetis_info_ls[1]

# Built locally
if not version:
    stream = os.popen("git")
