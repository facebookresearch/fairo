import a0
import collections.abc
import contextlib
import glob
import os
import pwd
import random
import shlex
import string
import subprocess
import sys


def fail(msg):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)


def common_env(proc_def):
    return dict(
        FBRP_NAME=proc_def.name,
        A0_TOPIC=proc_def.name,
        PYTHONUNBUFFERED="1",
    )


@contextlib.contextmanager
def common_env_context(proc_def):
    old_environ = dict(os.environ)
    os.environ.update(common_env(proc_def))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def stdout_logger():
    logger = a0.Logger(a0.env.topic())

    def write(msg):
        logger.info(msg)

    return write


def stderr_logger():
    logger = a0.Logger(a0.env.topic())

    def write(msg):
        logger.err(msg)

    return write


def pid_children(pid):
    for fname in glob.glob(f"/proc/{pid}/task/*/children"):
        with open(fname) as file:
            for line in file:
                yield int(line)


def is_ldap_user():
    user = pwd.getpwuid(os.getuid()).pw_name
    for line in open("/etc/passwd"):
        if line.startswith(f"{user}:"):
            return False
    return True


def nfs_root(path):
    result = subprocess.run(["findmnt", "-T", path], capture_output=True)
    lines = result.stdout.decode().split("\n")
    if len(lines) < 2:
        return None
    properties = dict(zip(lines[0].split(), lines[1].split()))
    if properties["FSTYPE"] != "nfs":
        return None
    return properties["TARGET"]


def random_string(alphabet=string.ascii_lowercase, length=16):
    return "".join(random.choices(alphabet, k=length))


# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def nested_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class NoEscape:
    def __init__(self, val):
        self.value = val


def shell_join(items):
    """Modified version of shlex.join that allows for non-escaped segments."""
    return " ".join(
        item.value if type(item) == NoEscape else shlex.quote(item) for item in items
    )
