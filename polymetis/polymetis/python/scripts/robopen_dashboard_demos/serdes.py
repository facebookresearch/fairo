import os
import site
import capnp
import threading

_schema_parser = capnp.SchemaParser()

capnp_cache = {}

lock = threading.Lock()


def get_capnp_msgs(msgpkg):
    with lock:
        if msgpkg not in capnp_cache:
            print(f"capnp loading {msgpkg}")
            filedir = os.path.dirname(os.path.abspath(__file__))
            capnp_cache[msgpkg] = _schema_parser.load(
                os.path.join(filedir, f"msgs/{msgpkg}.capnp"),
                imports=site.getsitepackages(),
            )
    return capnp_cache[msgpkg]
