import os
import json

import polymetis

polymetis_version = ""

if "CONDA_PREFIX" in os.environ and os.environ["CONDA_PREFIX"] in polymetis.__file__:
    # Conda installed
    stream = os.popen("conda list | grep polymetis")
    for line in stream:
        line_ls = [s for s in line.strip("\n").split(" ") if len(s) > 0]
        if line_ls[0] == "polymetis":
            polymetis_version = line_ls[1]
            break

else:
    # Built locally
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(polymetis.__file__))

    stream = os.popen("git describe --tags")
    version_strings = [line for line in stream][0].strip("\n").split("-")
    polymetis_version = f"{version_strings[-2]}_{version_strings[-1]}"

    os.chdir(original_cwd)

if not polymetis_version:
    raise Exception("Cannot locate Polymetis version!")
