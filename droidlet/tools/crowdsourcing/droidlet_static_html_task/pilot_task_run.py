#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

from mephisto.operations.operator import Operator
from mephisto.operations.utils import get_root_dir
from mephisto.abstractions.blueprints.static_html_task.static_html_blueprint import (
    BLUEPRINT_TYPE,
)
from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)

from mephisto.tools.scripts import load_db_and_process_config
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.qualification import QUAL_NOT_EXIST, make_qualification_dict

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Any

from pilot_config import PILOT_ALLOWLIST_QUAL_NAME, PILOT_BLOCK_QUAL_NAME, PILOT_QUAL_ANSWERS, KEYWORD_LIST

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    {"conf": "pilot_task"},
]

from mephisto.operations.hydra_config import RunScriptConfig, register_script_config

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

@dataclass
class TestScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY


register_script_config(name="scriptconfig", module=TestScriptConfig)


def validate_answers(answers, gt):
    freeform_answer = answers["freeform_answer"]
    commands = freeform_answer.splitlines()

    # filter empty commands
    filtered_commands = [x for x in commands if x != ""]

    # 1: Check that the number of commands >= 4
    if len(filtered_commands) < 4:
        return False

    # 2: Length check: Check that the average number of words in commands > 2
    commands_split = [x.split(" ") for x in filtered_commands]
    avg_words_in_commands = sum(map(len, commands_split)) / len(commands_split)
    if avg_words_in_commands <= 2:
        return False

    # 3: Keyword matching: Check that at least 2 keywords appear in the commands
    occurrence = sum([True if keyword in command.lower() else False for command in filtered_commands for keyword in KEYWORD_LIST])
    if occurrence < 2:
        return False

    # Validate multiple choice questions
    if answers["answer_0"] != gt["answer_0"] or \
       answers["answer_1"] != gt["answer_1"] or \
       answers["answer_2"] != gt["answer_2"]:
       return False

    return True

def validate_unit(unit):
    if unit.get_assigned_agent() is None:
        return
    output = mephisto_data_browser.get_data_from_unit(unit)["data"]
    if output is None:
        return

    answers = mephisto_data_browser.get_data_from_unit(unit)["data"]["outputs"]
    worker = unit.get_assigned_agent().get_worker()

    # Block every worker from working on this pilot task for second time
    # If they pass, they will be put into an allowlist
    # If they fail, they will never be able to see this again, unless configured explicitly
    try:
        db.make_qualification(PILOT_BLOCK_QUAL_NAME)
    except:
        pass
    else:
        logging.debug(f"{PILOT_BLOCK_QUAL_NAME} qualification not exists, so create one")
    worker.grant_qualification(PILOT_BLOCK_QUAL_NAME, 1)
    
    # Validate pilot task answers, workers who pass the validation will be put into an allowlist
    # by granting a qualification task to them and specify the task as a req in the full task
    try:
        db.make_qualification(PILOT_ALLOWLIST_QUAL_NAME)
    except:
        pass
    else:
        logging.debug(f"{PILOT_ALLOWLIST_QUAL_NAME} qualification not exists, so create one")

    if validate_answers(answers, PILOT_QUAL_ANSWERS):
        worker.grant_qualification(PILOT_ALLOWLIST_QUAL_NAME, 1)
        logging.info(f"Worker {worker.worker_name} passed the pilot task, put him/her into allowlist")

    return

@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)

    validator = validate_unit

    shared_state = SharedStaticTaskState(
        on_unit_submitted=validator,
    )
    # Do not allow workers to take pilot task the second time
    shared_state.qualifications = [
        make_qualification_dict(
            PILOT_BLOCK_QUAL_NAME,
            QUAL_NOT_EXIST,
            None,
        ),
    ]

    operator.validate_and_run_config(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
