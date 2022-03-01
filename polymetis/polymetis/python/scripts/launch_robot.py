#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import logging
import subprocess
import atexit
import sys
import time
import signal

import hydra

from polymetis.utils.data_dir import PKG_ROOT_DIR, which


log = logging.getLogger(__name__)


def check_server_exists(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        server_exists = s.connect_ex((ip, port)) == 0
    return server_exists


@hydra.main(config_name="launch_robot")
def main(cfg):
    build_dir = os.path.abspath(os.path.join(PKG_ROOT_DIR, "..", "..", "build"))
    log.info(f"Adding {build_dir} to $PATH")
    os.environ["PATH"] = build_dir + os.pathsep + os.environ["PATH"]

    # Check if another server is alive on address
    assert not check_server_exists(
        cfg.ip, cfg.port
    ), "Port unavailable; possibly another server found on designated address. To prevent undefined behavior, start the service on a different port or kill stale servers with 'pkill -9 run_server'"

    # Parse server address
    ip = str(cfg.ip)
    port = str(cfg.port)

    # Generate robot client config path (tmpfiles close implicitly upon gc)
    metadata_file = tempfile.NamedTemporaryFile(mode="wb")
    metadata = hydra.utils.instantiate(cfg.robot_client.metadata_cfg)
    metadata_file.write(metadata.serialize())
    metadata_file.flush()

    cfg_file = tempfile.NamedTemporaryFile(mode="w")
    cfg.robot_client.executable_cfg.robot_client_metadata_path = metadata_file.name
    cfg_pretty = OmegaConf.to_yaml(cfg.robot_client.executable_cfg, resolve=True)
    log.info(f"=== Config: ===\n{cfg_pretty}")
    cfg_file.write(cfg_pretty)
    cfg_file.flush()

    # Start server
    log.info(f"Starting server")
    server_exec_path = which(cfg.server_exec)
    server_cmd = [server_exec_path]
    server_cmd = server_cmd + ["-s", ip, "-p", port, "-c", cfg_file.name]

    if cfg.use_real_time:
        log.info(f"Acquiring sudo...")
        subprocess.run(["sudo", "echo", '"Acquired sudo."'], check=True)

        server_cmd = ["sudo", "-s", "env", '"PATH=$PATH"'] + server_cmd + ["-r"]
    server_output = subprocess.Popen(
        server_cmd, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setpgrp
    )
    pgid = os.getpgid(server_output.pid)

    # Kill process at the end
    if cfg.use_real_time:

        def cleanup():
            log.info(
                f"Using sudo to kill subprocess with pid {server_output.pid}, pgid {pgid}..."
            )
            # send NEGATIVE of process group ID to kill process tree
            subprocess.check_call(["sudo", "kill", "-9", f"-{pgid}"])

    else:

        def cleanup():
            log.info(f"Killing subprocess with pid {server_output.pid}, pgid {pgid}...")
            subprocess.check_call(["kill", "-9", f"-{pgid}"])

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signal_number, stack_frame: cleanup())


if __name__ == "__main__":
    main()
