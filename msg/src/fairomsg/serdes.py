import os
import site
import threading
import glob

import capnp

_schema_parser = capnp.SchemaParser()
capnp_cache = {}
lock = threading.Lock()

def _get_full_filepath(msgpkg):
    filedir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(filedir, f"def/{msgpkg}.capnp")


def get_msgs(msgpkg):
    with lock:
        if msgpkg not in capnp_cache:
            print(f"capnp loading {msgpkg}")
            capnp_cache[msgpkg] = _schema_parser.load(_get_full_filepath(msgpkg), imports=site.getsitepackages()
            )
    return capnp_cache[msgpkg]

def get_pkgs():
    filedir = os.path.dirname(os.path.abspath(__file__))
    filenames = glob.glob(f"{filedir}/def/*.capnp")
    return [os.path.splitext(os.path.basename(filename))[0] for filename in filenames]
