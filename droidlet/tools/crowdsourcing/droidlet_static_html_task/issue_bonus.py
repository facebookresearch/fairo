#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import json
import boto3
import csv
import os

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from mephisto.data_model.worker import Worker

logging.basicConfig(level="INFO")

db = LocalMephistoDB()
data_browser = DataBrowser(db=db)

s3 = boto3.client("s3")


def issue_bonuses(task_name: str) -> list:
    logging.info(f"Initializing bonus script for Mephisto task_name: {task_name}")

    # Download the shared list of issued bonuses and pull out unique reference tuples to check against
    logging.info(f"Downloading interaction bonus records from S3...")
    with open("bonus_records.csv", "wb") as f:
        s3.download_fileobj("droidlet-hitl", "bonus_records.csv", f)

    logging.info(f"Building list of already issued bonuses...")
    previously_issued_units = []
    with open("bonus_records.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            previously_issued_units.append(
                (row[0], row[1])
            )  # the combination of task_name and unit_id is essentially unique

    # Get completed units from the run_id
    logging.info(f"Retrieving units from Mephisto based on task_name...")
    units = data_browser.get_units_for_task_name(task_name)
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_units.append(unit)

    logging.info(f"Completed units for job {task_name} retrieved")

    # Retrieve bonus info from DB and issue
    new_bonus_records = []
    bonus_results = []
    total_bonus = 0
    units_skipped = 0
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        unit_id = data["unit_id"]
        if (task_name, unit_id) not in previously_issued_units:
            worker = Worker(db, data["worker_id"])
            outputs = data["data"]["outputs"]
            clean_click_string = outputs["clickedElements"].replace("'", "")
            clicks = json.loads(clean_click_string)
            bonus_result = False
            if clicks:
                for click in clicks:
                    if "interactionScores" in click["id"]:
                        try:
                            amount = float(
                                f'{(click["id"]["interactionScores"]["stoplight"] * 0.30):.2f}'
                            )
                            bonus_result, _ = worker.bonus_worker(
                                amount, "Virtual assistant interaction quality bonus", unit
                            )
                            total_bonus += amount
                            new_bonus_records.append(
                                (task_name, unit_id, worker.worker_name, amount)
                            )
                        except:
                            logging.error(
                                f"Exception raised on bonus issue for {worker.worker_name}, debug"
                            )
                            new_bonus_records.append(
                                (task_name, unit_id, worker.worker_name, "ERR")
                            )
                            pass
                if not bonus_result:
                    logging.info(
                        f"Bonus NOT successfully issued for worker {worker.worker_name}, but no error was raised.  \
                        Make sure interaction score exists and retry."
                    )
            else:
                logging.info(
                    f"Recorded click data not found for {worker.worker_name}, no bonus will be issued"
                )
            bonus_results.append(bonus_result)
        else:
            units_skipped += 1

    logging.info(f"Num completed units: {len(completed_units)}")
    logging.info(
        f"Num bonuses skipped because bonus was issued previously for the same unit: {units_skipped}"
    )
    logging.info(f"Num new bonuses issued: {len([x for x in bonus_results if x])}")
    logging.info(f"Num bonuses FAILED: {len([x for x in bonus_results if not x])}")
    logging.info(f"Total bonus amount issued: {total_bonus}")

    if new_bonus_records:
        logging.info(f"There are newly issued bonuses to record")
        logging.info(f"Writing new bonuses to csv and uploading to S3...")
        with open("bonus_records.csv", "a") as f:
            writer = csv.writer(f)
            for record in new_bonus_records:
                writer.writerow(record)
        s3.upload_file("bonus_records.csv", "droidlet-hitl", "bonus_records.csv")

    os.remove("bonus_records.csv")
    logging.info(f"Finished issuing bonuses!")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, help="Mephisto task name", required=True)
    args = parser.parse_args()

    issue_bonuses(args.task_name)
