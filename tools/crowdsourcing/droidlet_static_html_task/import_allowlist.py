#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

from mephisto.operations.operator import Operator
from mephisto.data_model.worker import Worker
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

from pilot_config import PILOT_ALLOWLIST_QUAL_NAME, PILOT_BLOCK_QUAL_NAME, PILOT_QUAL_ANSWERS, KEYWORD_LIST

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Any

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)


def importAllowlist():
    wd = os.path.dirname(os.path.abspath(__file__))
    turkerListFilepath = os.path.join(wd, 'turker_list.txt')
    listfile = open(turkerListFilepath, "r")
    allowlist = [line.strip() for line in listfile.readlines()]
    listfile.close()

    return allowlist


def addQual(worker):
    # Block every worker from working on this pilot task for second time
    try:
        db.make_qualification(PILOT_BLOCK_QUAL_NAME)
    except:
        pass
    else:
        logging.debug(f"{PILOT_BLOCK_QUAL_NAME} qualification not exists, so create one")
    worker.grant_qualification(PILOT_BLOCK_QUAL_NAME, 1)
    
    # Workers who pass the validation will be put into an allowlist
    try:
        db.make_qualification(PILOT_ALLOWLIST_QUAL_NAME)
    except:
        pass
    else:
        logging.debug(f"{PILOT_ALLOWLIST_QUAL_NAME} qualification not exists, so create one")

    worker.grant_qualification(PILOT_ALLOWLIST_QUAL_NAME, 1)
    logging.info(f"Worker {worker.worker_name} passed the pilot task, put him/her into allowlist")

    return


def main():
    allowlist = importAllowlist()

    for turker in allowlist:
        #First add the worker to the database, or retrieve them if they already exist
        try:
            db_id = db.new_worker(turker, 'mturk')
            worker = Worker.get(db, db_id)
        except:
            worker = db.find_workers(turker, 'mturk')[0]
        #Then add them to the qualification
        addQual(worker)
        #Check to make sure the qualification was added successfully
        if not worker.is_qualified(PILOT_ALLOWLIST_QUAL_NAME):
            logging.info(f"!!!Worker not successfully qualified, debug")
        else:
            logging.info(f"Worker successfully qualified")

    return

if __name__ == "__main__":
    main()