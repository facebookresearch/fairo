#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from mephisto.abstractions.blueprints.static_html_task.static_html_blueprint import (
    BLUEPRINT_TYPE_STATIC_HTML,
)
from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)
from mephisto.data_model.qualification import QUAL_EXISTS, QUAL_NOT_EXIST
from mephisto.utils.qualifications import make_qualification_dict
from pilot_config import PILOT_ALLOWLIST_QUAL_NAME as ALLOWLIST_QUALIFICATION
from pilot_config import SOFTBLOCK_QUAL_NAME as SOFTBLOCK_QUALIFICATION

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Any

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    "_self_",
    {"mephisto/blueprint": BLUEPRINT_TYPE_STATIC_HTML},
    {"mephisto/architect": "ec2"},
    {"mephisto/provider": "mock"},
    {"conf": "annotation"},
]

from mephisto.operations.hydra_config import RunScriptConfig, register_script_config


@dataclass
class TestScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY


register_script_config(name="scriptconfig", module=TestScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    shared_state = SharedStaticTaskState(
        qualifications=[
            make_qualification_dict(ALLOWLIST_QUALIFICATION, QUAL_EXISTS, None),
            make_qualification_dict(SOFTBLOCK_QUALIFICATION, QUAL_NOT_EXIST, None),
        ],
    )

    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)

    operator.validate_and_run_config(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
