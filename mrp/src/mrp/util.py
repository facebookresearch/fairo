import a0
import asyncio
import collections.abc
import contextlib
import glob
import os
import pwd
import random
import shlex
import string
import subprocess


def common_env(proc_def):
    return dict(
        MRP_NAME=proc_def.name,
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


class PlainTextLogger:
    def __init__(self):
        self._logger = a0.Logger(a0.env.topic())

    def _plaintext_pkt(self, payload=None, headers=None):
        payload = payload or ""
        headers = headers or []
        headers.append(("content-type", "text/plain"))
        return a0.Packet(headers, payload)

    def info(self, payload=None, *, headers=None):
        self._logger.info(self._plaintext_pkt(payload, headers))

    def err(self, payload=None, *, headers=None):
        self._logger.err(self._plaintext_pkt(payload, headers))


class LogPtyPipes:
    def __init__(self, fd_in, fd_out, fd_err):
        self.fd_in = fd_in
        self.fd_out = fd_out
        self.fd_err = fd_err
        self.ptl = PlainTextLogger()

    def start(self):
        loop = asyncio.get_event_loop()
        loop.add_reader(
            self.fd_out,
            self._read_impl,
            self.fd_out,
            a0.Publisher(f"mrp/{a0.env.topic()}/stdout").pub,
            self.ptl.info,
            [b""],
        )
        loop.add_reader(
            self.fd_err,
            self._read_impl,
            self.fd_err,
            a0.Publisher(f"mrp/{a0.env.topic()}/stderr").pub,
            self.ptl.err,
            [b""],
        )
        self._write_task = asyncio.create_task(self._write_impl())

    async def stop(self):
        loop = asyncio.get_event_loop()
        loop.remove_reader(self.fd_out)
        loop.remove_reader(self.fd_err)
        self._write_task.cancel()

    def _read_impl(self, fd, pub, log, residual):
        data = os.read(fd, 4 * 1024)
        if not data:
            return

        pub(data)

        lines = data.split(b"\n")
        if len(lines) == 1:
            residual[0] += lines[0]
            return

        lines[0] = residual[0] + lines[0]

        for line in lines[:-1]:
            log(line[: 4 * 1024])

        residual[0] = lines[-1]

    async def _write_impl(self):
        async for pkt in a0.aio_sub(f"mrp/{a0.env.topic()}/stdin"):
            os.write(self.fd_in, pkt.payload)


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
