#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import json

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from mephisto.data_model.worker import Worker

logging.basicConfig(level="INFO")

db = LocalMephistoDB()
data_browser = DataBrowser(db=db)

def main(run_id) -> list:
    logging.info(f"Initializing bonus script for Mephisto job {run_id}")

    # Get completed units from the run_id
    units = db.find_units(task_run_id=run_id)
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_units.append(unit)

    logging.info(f"Completed units for job {run_id} retrieved")

    # Retrieve bonus info from DB and issue
    bonus_results = []
    total_bonus = 0
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        worker = Worker(db, data["worker_id"])
        outputs = data["data"]["outputs"]
        clean_click_string = outputs["clickedElements"].replace("'", "")
        clicks = json.loads(clean_click_string)
        if clicks:
            for click in clicks:
                if "interactionScores" in click["id"]:
                    amount = float(f'{(click["id"]["interactionScores"]["stoplight"] * 0.30):.2f}')
                    total_bonus += amount
                    bonus_result, _ = worker.bonus_worker(amount, "Virtual assistant interaction quality bonus", unit)
                    bonus_results.append(bonus_result)
                    if not bonus_result:
                        logging.info(f"Bonus NOT successfully issued for worker {worker.worker_name} , debug")
        else:
            logging.info(f'Recorded click data not found for {worker.worker_name}, no bonus will be issued')

    logging.info(f"Num completed units: {len(completed_units)}")
    logging.info(f"Num bonuses issued: {len([x for x in bonus_results if x])}")
    logging.info(f"Total bonus amount issued: {total_bonus}")

    return bonus_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, help="integer mephisto run ID", required=True)
    args = parser.parse_args()

    main(args.run_id)