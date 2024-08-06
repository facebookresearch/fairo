#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from mephisto.tools.scripts import task_script
from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)
from mephisto.data_model.qualification import QUAL_EXISTS, QUAL_NOT_EXIST
from mephisto.utils.qualifications import make_qualification_dict

from omegaconf import DictConfig

from droidlet.tools.crowdsourcing.droidlet_static_html_task.pilot_config import (
    PILOT_ALLOWLIST_QUAL_NAME as ALLOWLIST_QUALIFICATION,
)
from droidlet.tools.crowdsourcing.droidlet_static_html_task.pilot_config import (
    SOFTBLOCK_QUAL_NAME as SOFTBLOCK_QUALIFICATION,
)


@task_script(default_config_file="run_with_qual")
def main(operator: Operator, cfg: DictConfig) -> None:
    shared_state = SharedStaticTaskState(
        qualifications=[
            make_qualification_dict(ALLOWLIST_QUALIFICATION, QUAL_EXISTS, None),
            make_qualification_dict(SOFTBLOCK_QUALIFICATION, QUAL_NOT_EXIST, None),
        ],
    )

    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)

    operator.launch_task_run(cfg.mephisto, shared_state)
    # operator.launch_task_run(cfg.mephisto)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
