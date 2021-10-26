#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from mephisto.operations.operator import Operator
from mephisto.operations.utils import get_root_dir
from mephisto.tools.scripts import load_db_and_process_config
from mephisto.abstractions.blueprints.static_html_task.static_html_blueprint import (
    BLUEPRINT_TYPE,
)
from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)
from mephisto.data_model.qualification import QUAL_EXISTS, QUAL_NOT_EXIST, make_qualification_dict

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Any

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "heroku"},
    {"mephisto/provider": "mock"},
    {"conf": "run_with_qual"},
]

from mephisto.operations.hydra_config import RunScriptConfig, register_script_config

ALLOWLIST_QUALIFICATION = "PILOT_ALLOWLIST_QUAL_0920_0"


@dataclass
class TestScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY


register_script_config(name="scriptconfig", module=TestScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:

    def onboarding_is_valid(onboarding_data):
        outputs = onboarding_data["outputs"]
        answer_str = outputs["answer"]
        # NOTE: depending on which OS Turker uses, there could be carriage returns \r or just newlines \n
        # this python module should handle all cases
        commands = answer_str.splitlines()
        # filter empty commands
        filtered_commands = [x for x in commands if x != ""]
        # Number check: Check that the number of commands >= 3
        if len(commands) < 3:
            return False
        # Length check: Check that the average number of words in commands > 4
        commands_split = [x.split(" ") for x in filtered_commands]
        avg_words_in_commands = sum(map(len, commands_split)) / len(commands_split)
        if avg_words_in_commands < 2:
            return False
        # Diversity check: Check that commands are reasonably diverse
        first_words = [x[0] for x in commands_split]
        if len(set(first_words)) == 1:
            return False
        # TODO: Grammar check: Check that there is punctuation, capitals
        return True

    shared_state = SharedStaticTaskState(
        make_qualification_dict(
            ALLOWLIST_QUALIFICATION,
            QUAL_EXISTS,
            None
        ),
    )

    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)

    operator.validate_and_run_config(cfg.mephisto, shared_state)
    # operator.validate_and_run_config(cfg.mephisto)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()