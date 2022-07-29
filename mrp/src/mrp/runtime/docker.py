from mrp import life_cycle
from mrp import util
from mrp.process_def import ProcDef
from mrp.runtime.base import BaseLauncher, BaseRuntime
import a0
import aiodocker
import aiohttp
import asyncio
import contextlib
import docker
import io
import json
import os
import pathlib
import pwd
import typing


class Launcher(BaseLauncher):
    def __init__(
        self,
        image,
        mount: typing.List[str],
        name: str,
        proc_def: ProcDef,
        **kwargs,
    ):
        self.image = image
        self.mount = mount
        self.name = name
        self.proc_def = proc_def
        self.kwargs = kwargs

    async def run(self):
        container = f"mrp_{self.name}"

        # Environmental variables.
        env = util.common_env(self.proc_def)
        for kv in self.kwargs.pop("Env", []):
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
            else:
                del env[kv]
        env.update(self.proc_def.env)

        # Docker labels.
        labels = {
            "mrp.name": self.name,
            "mrp.container": container,
        }
        labels.update(self.kwargs.pop("labels", {}))

        # Mount volumes.
        for f in [
            "group",
            "gshadow",
            "inputrc",
            "localtime",
            "passwd",
            "shadow",
            "subgid",
            "subuid",
            "sudoers",
        ]:
            self.mount.append(f"/etc/{f}:/etc/{f}:ro")

        self.mount.append("/var/lib/sss:/var/lib/sss:ro")

        id_info = pwd.getpwuid(os.getuid())

        # Compose all the arguments.
        run_kwargs = {
            "Image": self.image,
            "Env": ["=".join(kv) for kv in env.items()],
            "Labels": labels,
            "User": f"{id_info.pw_uid}:{id_info.pw_gid}",
            "Tty": True,
            "OpenStdin": True,
            "HostConfig": {
                "Privileged": True,
                "NetworkMode": "host",
                "PidMode": "host",
                "IpcMode": "host",
                "Binds": self.mount,
                "GroupAdd": [str(i) for i in os.getgroups()],
            },
        }
        util.nested_dict_update(run_kwargs, self.kwargs)

        docker = aiodocker.Docker()

        try:
            self.proc = await docker.containers.create_or_replace(container, run_kwargs)
            self.proc_io = await self.proc.websocket()
            await self.proc.start()
            proc_info = await self.proc.show()
        except Exception as e:
            life_cycle.set_state(
                self.name, life_cycle.State.STOPPED, return_code=-1, error_info=str(e)
            )
            await docker.close()
            raise RuntimeError(
                f"Failed to start docker process: name={self.name} reason={e}"
            )

        self.proc_pid = proc_info["State"]["Pid"]
        life_cycle.set_state(self.name, life_cycle.State.STARTED)

        ptl = util.PlainTextLogger()

        async def log_pipe(log_fn, pipe):
            async for line in pipe:
                log_fn(line)

        self.down_task = asyncio.create_task(self.down_watcher(self.handle_down))
        try:
            await asyncio.gather(
                log_pipe(ptl.info, self.proc.log(stdout=True, follow=True)),
                log_pipe(ptl.err, self.proc.log(stderr=True, follow=True)),
                self.stdout_pipe(),
                self.stdin_pipe(),
                self.log_psutil(),
                self.death_handler(),
                self.down_task,
            )
        except asyncio.exceptions.CancelledError:
            # death_handler cancelled down listener.
            pass
        await docker.close()

    def get_pid(self):
        return self.proc_pid

    async def death_handler(self):
        result = await self.proc.wait()
        life_cycle.set_state(
            self.name, life_cycle.State.STOPPED, return_code=result["StatusCode"]
        )
        # TODO(lshamis): Restart policy goes here.

        # Release the down listener.
        self.down_task.cancel()

    async def handle_down(self):
        await self.proc.stop()
        with contextlib.suppress(asyncio.TimeoutError):
            await self.proc.wait(timeout=3.0)
        await self.proc.delete(force=True)

    async def stdout_pipe(self):
        pub = a0.Publisher(f"mrp/{a0.env.topic()}/stdout").pub

        while True:
            msg = await self.proc_io.receive()
            if msg.type not in [aiohttp.WSMsgType.BINARY, aiohttp.WSMsgType.TEXT]:
                break
            pub(msg.data)

    async def stdin_pipe(self):
        try:
            async for pkt in a0.aio_sub(f"mrp/{a0.env.topic()}/stdin"):
                await self.proc_io.send_bytes(pkt.payload)
        except ConnectionResetError:
            pass


class Docker(BaseRuntime):
    def __init__(
        self,
        image=None,
        dockerfile=None,
        mount=None,
        build_kwargs=None,
        run_kwargs=None,
    ):
        mount = mount or []
        build_kwargs = build_kwargs or {}
        run_kwargs = run_kwargs or {}

        if bool(image) == bool(dockerfile):
            raise ValueError(
                "Docker process must define exactly one of image or dockerfile"
            )
        self.image = image
        self.dockerfile = dockerfile
        self.mount = mount
        self.build_kwargs = build_kwargs
        self.run_kwargs = run_kwargs

    def asdict(self, root: pathlib.Path):
        ret = {}
        if self.image:
            ret["image"] = self.image
        if self.dockerfile:
            ret["dockerfile"] = self.dockerfile
        if self.mount:
            ret["mount"] = self.mount
        if self.build_kwargs:
            ret["build_kwargs"] = self.build_kwargs
        if self.run_kwargs:
            ret["run_kwargs"] = self.run_kwargs
        return ret

    def _build(self, name: str, proc_def: ProcDef, cache: bool, verbose: bool):
        docker_api = docker.from_env()
        docker_api.lowlevel = docker.APIClient()

        if self.dockerfile:
            self.image = f"mrp/{name}"

            dockerfile_path = os.path.join(
                os.path.dirname(proc_def.rule_file), self.dockerfile
            )

            build_kwargs = {
                "tag": self.image,
                "path": proc_def.root,
                "dockerfile": dockerfile_path,
                "rm": True,
                "nocache": not cache,
            }
            build_kwargs.update(self.build_kwargs)

            for line in docker_api.lowlevel.build(**build_kwargs):
                lineinfo = json.loads(line.decode())
                if verbose and "stream" in lineinfo:
                    print(lineinfo["stream"].strip())
                elif "errorDetail" in lineinfo:
                    raise RuntimeError(json.dumps(lineinfo["errorDetail"], indent=2))
        else:
            should_pull = not cache
            if cache:
                try:
                    docker_api.images.get(self.image)
                except docker.errors.ImageNotFound:
                    should_pull = True

            if should_pull:
                try:
                    for line in docker_api.lowlevel.pull(self.image, stream=True):
                        lineinfo = json.loads(line.decode())
                        if verbose and "status" in lineinfo:
                            print(lineinfo["status"].strip())
                except docker.errors.NotFound as e:
                    raise RuntimeError(e)

        uses_ldap = util.is_ldap_user()

        mount_map = {}
        for mnt in self.mount:
            parts = mnt.split(":")
            if len(parts) == 3:
                mount_map[parts[0]] = (parts[1], parts[2])
            elif len(parts) == 2:
                mount_map[parts[0]] = (parts[1], "")
            else:
                raise ValueError(f"Invalid mount: {mnt}")

        nfs_mounts = []
        for host, (container, _) in mount_map.items():
            try:
                os.makedirs(host)
            except FileExistsError:
                pass
            except FileNotFoundError:
                pass

            nfs_root = util.nfs_root(host)
            if nfs_root and nfs_root != host:
                nfs_mounts.append(
                    {
                        "host_nfs_root": nfs_root,
                        "container_nfs_root": f"/mrp_nfs/{util.random_string()}",
                        "host": host,
                        "container": container,
                    }
                )

        for nfs_mount in nfs_mounts:
            if nfs_mount["host"] in mount_map:
                del mount_map[nfs_mount["host"]]

        if uses_ldap or nfs_mounts:
            dockerfile = [f"FROM {self.image}"]
            if uses_ldap:
                # We try to inject sssd to allow the container to communicate with the ldap server.
                # We assume the docker container provides apt. If there is no apt, we skip this step. The user id and group ids will still correctly match the active user, but the user and group names will not be fetchable.
                apt_install_sssd_cmd = " && ".join(
                    [
                        "apt update",
                        "DEBIAN_FRONTEND=noninteractive apt install -y sssd",
                        "rm -rf /var/lib/apt/lists/*",
                    ]
                )
                try_apt_install_sssd_cmd = (
                    f"if [ $(which apt) ] ; then $({apt_install_sssd_cmd}) ; fi"
                )
                dockerfile.append(f"RUN {try_apt_install_sssd_cmd}")
            for nfs_mount in nfs_mounts:
                relpath = os.path.relpath(nfs_mount["host"], nfs_mount["host_nfs_root"])
                container_fullpath = os.path.join(
                    nfs_mount["container_nfs_root"], relpath
                )
                mkdir = f"mkdir -p {os.path.dirname(nfs_mount['container'])}"
                link = f"ln -s {container_fullpath} {nfs_mount['container']}"
                dockerfile.append(f"RUN {mkdir} && {link}")

                mount_map[nfs_mount["host_nfs_root"]] = (
                    nfs_mount["container_nfs_root"],
                    "",
                )

            self.image = f"mrp/{name}:netext"

            for line in docker_api.lowlevel.build(
                fileobj=io.BytesIO("\n".join(dockerfile).encode("utf-8")),
                rm=True,
                tag=self.image,
                labels={
                    "mrp.type": "netext",
                    "mrp.tmp": "1",
                },
            ):
                lineinfo = json.loads(line.decode())
                if verbose and "stream" in lineinfo:
                    print(lineinfo["stream"].strip())
                elif "errorDetail" in lineinfo:
                    raise RuntimeError(json.dumps(lineinfo["errorDetail"], indent=2))

        self.mount = [
            f"{host_path}:{container_path}:{options}"
            for host_path, (container_path, options) in mount_map.items()
        ]

    def _launcher(self, name: str, proc_def: ProcDef):
        return Launcher(self.image, self.mount, name, proc_def, **self.run_kwargs)
