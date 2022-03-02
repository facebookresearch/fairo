# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import tempfile
import subprocess
import sys

import omegaconf
from omegaconf import OmegaConf

from polymetis.utils.data_dir import which
from polymetis.robot_client.abstract_robot_client import (
    AbstractRobotClient,
)

log = logging.getLogger(__name__)


class ExecutableRobotClient(AbstractRobotClient):
    """A RobotClient which calls some executable as a subprocess.

    This RobotClient is used for instantiating some executable as a subprocess,
    e.g. a connection to a robot requiring a real-time control loop.

    Args:
        executable_cfg: A Hydra configuration object detailing the executable.
                        e.g. `../conf/robot_client/empty_statistics_client`.

        use_real_time: If `True`, this will call the executable with sudo.

    """

    def __init__(
        self, executable_cfg: omegaconf.DictConfig, use_real_time: bool, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_real_time = use_real_time
        self.executable_cfg = executable_cfg

    def run(self):
        """Connects to gRPC server, and passes executable client required files.

        Note:
            Creates two temporary files: an executable configuration file,
            and a metadata protobuf message. The configuration file contains
            the path to the metadata file. The path of the configuration is
            passed to the executable.

        """
        with tempfile.NamedTemporaryFile(mode="w") as cfg_file:
            with tempfile.NamedTemporaryFile(mode="wb") as metadata_file:
                # Write metadata to temp file
                metadata_file.write(self.metadata.serialize())
                metadata_file.flush()

                # Save metadata file path
                self.executable_cfg.robot_client_metadata_path = metadata_file.name

                # Write configuration to temporary file
                cfg_pretty = OmegaConf.to_yaml(self.executable_cfg, resolve=True)
                log.info(f"=== Config: ===\n{cfg_pretty}")
                cfg_file.write(cfg_pretty)
                cfg_file.flush()

                # Find path to executable
                path_to_exec = which(self.executable_cfg.exec)
                assert path_to_exec, f"Unable to find binary {self.executable_cfg.exec}"

                # Add sudo if realtime; also, inherit $PATH variable
                command_list = [path_to_exec, cfg_file.name]
                if self.use_real_time:
                    command_list = ["sudo", "env", '"PATH=$PATH"'] + command_list

                # Run
                log.info(f"=== Executing client at {path_to_exec} ===")
                subprocess.run(
                    command_list,
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    check=True,
                )
