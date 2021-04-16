"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import hashlib
import hooks


HOOK_MAP = {getattr(hooks, h): h for h in dir(hooks) if not h.startswith("__")}


def get_hook_name(hook_id):
    return HOOK_MAP[hook_id]


def get_hashes(path):
    with open(path, "rb") as f:
        contents = f.read()
    raw_hash = hashlib.sha1(contents).hexdigest()
    # Cuberite sometimes (?) rewrites .ini files with CRLF
    nocr_hash = hashlib.sha1(contents.replace(b"\x0d\x0a", b"\x0a")).hexdigest()
    return (raw_hash, nocr_hash)
