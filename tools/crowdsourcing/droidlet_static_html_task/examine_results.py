#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker
from mephisto.data_model.unit import Unit

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

DO_REVIEW = True

units = mephisto_data_browser.get_units_for_task_name(input("Input task name: "))

tasks_to_show = input("Tasks to see? (a)ll/(u)nreviewed: ")
if tasks_to_show in ["all", "a"]:
    DO_REVIEW = False
else:
    units = [u for u in units if u.get_status() == "completed"]
    print(
        "You will be reviewing actual tasks with this flow. Tasks that you either Accept or Pass "
        "will be paid out to the worker, while rejected tasks will not. Passed tasks will be "
        "specially marked such that you can leave them out of your dataset. \n"
        "When you pass on a task, the script gives you an option to disqualify the worker "
        "from future tasks by assigning a qualification. If provided, this worker will no "
        "longer be able to work on tasks where the set --block-qualification shares the same name.\n"
        "You should only reject tasks when it is clear the worker has acted in bad faith, and "
        "didn't actually do the task. Prefer to pass on tasks that were misunderstandings."
    )


def format_for_printing_data(data):
    # Custom tasks can define methods for how to display their data in a relevant way
    worker_name = Worker(db, data["worker_id"]).worker_name
    contents = data["data"]
    duration = contents["times"]["task_end"] - contents["times"]["task_start"]
    metadata_string = (
        f"Worker: {worker_name}\nUnit: {data['unit_id']}\n"
        f"Duration: {int(duration)}\nStatus: {data['status']}\n"
    )

    inputs = contents["inputs"]
    inputs_string = f"Domain: {inputs['subdomain']}\n"

    outputs = contents["outputs"]
    output_string = ""
    try:
        output_string += f"Usability Rating: {outputs['usability-rating']}\n"
    except:
        pass
    try:
        output_string += f"Self Performance Rating: {outputs['self-rating']}\n"
    except:
        pass
    try:
        output_string += f"Instructions Read Time (sec): {outputs['instructionsReadTime']}\n"
    except:
        pass
    try:
        output_string += f"Pre Interaction Time (sec): {outputs['preInteractTime']}\n"
    except:
        pass
    try:
        output_string += f"Interaction Time (sec): {outputs['interactTime']}\n"
    except:
        pass
    try:
        output_string += f"Clicks (timestamp): {outputs['clickedElements']}\n"
    except:
        pass
    #found_files = outputs.get("files")
    #if found_files is not None:
    #    file_dir = Unit(db, data["unit_id"]).get_assigned_agent().get_data_dir()
    #    output_string += f"   Files: {found_files}\n"
    #    output_string += f"   File directory {file_dir}\n"
    #else:
    #    output_string += f"   Files: No files attached\n"
    return f"-------------------\n{metadata_string}{inputs_string}{output_string}"


disqualification_name = None
for unit in units:
    print(format_for_printing_data(mephisto_data_browser.get_data_from_unit(unit)))
    if DO_REVIEW:
        keep = input("Do you want to accept this work? (a)ccept, (r)eject, (p)ass: ")
        if keep == "a":
            unit.get_assigned_agent().approve_work()
        elif keep == "r":
            reason = input("Why are you rejecting this work?")
            unit.get_assigned_agent().reject_work(reason)
        elif keep == "p":
            # General best practice is to accept borderline work and then disqualify
            # the worker from working on more of these tasks
            agent = unit.get_assigned_agent()
            agent.soft_reject_work()
            should_soft_block = input("Do you want to soft block this worker? (y)es/(n)o: ")
            if should_soft_block.lower() in ["y", "yes"]:
                if disqualification_name == None:
                    disqualification_name = input(
                        "Please input the qualification name you are using to soft block for this task: "
                    )
                worker = agent.get_worker()
                worker.grant_qualification(disqualification_name, 1)
