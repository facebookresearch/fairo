import logging
import tempfile
import subprocess
import sys
import os

import omegaconf
from omegaconf import OmegaConf

from polymetis.utils.data_dir import PKG_ROOT_DIR, which
from polymetis.robot_client.abstract_robot_client import (
    AbstractRobotClient,
)

log = logging.getLogger(__name__)


class ExecutableRobotServer:
    """An object which calls some executable as a subprocess.

    Args:
        cfg: A Hydra configuration object to be passed into the executable
             in the form of a path to a temporary YAML.
    """

    def __init__(self, cfg, exec):
        self.cfg = cfg

        # Add build dir to path
        build_dir = os.path.abspath(os.path.join(PKG_ROOT_DIR, "..", "..", "build"))
        log.info(f"Adding {build_dir} to $PATH")
        os.environ["PATH"] = build_dir + os.pathsep + os.environ["PATH"]

        # Find path to executable
        self.path_to_exec = which(exec)
        assert self.path_to_exec, f"Unable to find binary {exec}"

    def run(self):
        with tempfile.NamedTemporaryFile(mode="w") as cfg_file:
            # Write configuration to temporary file
            cfg_pretty = OmegaConf.to_yaml(self.cfg, resolve=True)
            log.info(f"=== Config: ===\n{cfg_pretty}")
            cfg_file.write(cfg_pretty)
            cfg_file.flush()

            # Run
            log.info(f"=== Executing {self.path_to_exec} ===")
            command_list = [self.path_to_exec, cfg_file.name]
            subprocess.run(
                command_list,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
