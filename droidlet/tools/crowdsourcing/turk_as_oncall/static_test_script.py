#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.operations.operator import Operator
from mephisto.tools.scripts import task_script
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig


@task_script(default_config_file="example")
def main(operator: Operator, cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)

    operator.launch_task_run(cfg.mephisto)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
