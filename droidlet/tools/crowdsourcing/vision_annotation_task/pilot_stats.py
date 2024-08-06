# %%
"""
Get stats and plot for vision annotation pilot tasks
"""

# %%
from numpy import Inf, Infinity
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Tuple
from datetime import datetime

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from mephisto.data_model.worker import Worker


# %%
def check_run_status(run_id: int, qual_name: str) -> None:
    db = LocalMephistoDB()
    units = db.find_units(task_run_id=run_id)
    units_num = len(units)
    completed_num = 0
    launched_num = 0
    assigned_num = 0
    accepted_num = 0
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_num += 1
            completed_units.append(unit)
        elif unit.db_status == "launched":
            launched_num += 1
        elif unit.db_status == "assigned":
            assigned_num += 1
        elif unit.db_status == "accepted":
            accepted_num += 1
    print(
        f"Total HIT num: {units_num}\tCompleted HIT num: {completed_num}\tCompleted rate: {completed_num / units_num * 100}%"
    )
    print(
        f"Total HIT num: {units_num}\tLaunched HIT num: {launched_num}\tLaunched rate: {launched_num / units_num * 100}%"
    )
    print(
        f"Total HIT num: {units_num}\tAssigned HIT num: {assigned_num}\tAssigned rate: {assigned_num / units_num * 100}%"
    )
    print(
        f"Total HIT num: {units_num}\tAccepted HIT num: {accepted_num}\tAccepted rate: {accepted_num / units_num * 100}%"
    )

    data_browser = DataBrowser(db=db)
    total_time_completed_in_min = 0
    total_cnt = 0
    passed_time = 0
    passed_cnt = 0
    turkers_with_mturk_qual_cnt = 0
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        duration = (
            data["data"]["times"]["task_end"] - data["data"]["times"]["task_start"]
        ) / 60  # in minutes
        if duration > 0:
            total_time_completed_in_min += duration
            total_cnt += 1
        worker_name = db.get_worker(worker_id=unit.worker_id)["worker_name"]
        turkers_with_mturk_qual_cnt += 1
        worker = db.find_workers(worker_name=worker_name)[0]
        if worker.get_granted_qualification(qual_name):
            passed_time += duration
            passed_cnt += 1

    print(
        f"For mephisto/mturk debug: total num: {total_cnt}, # who pass mturk qual: {turkers_with_mturk_qual_cnt}"
    )
    print(
        f"Total completed HITS\t\t{total_cnt}\tavg time spent\t{total_time_completed_in_min / total_cnt} mins"
    )
    print(
        f"HITS passed qualification\t{passed_cnt}\tavg time spent\t{passed_time / passed_cnt} mins"
    )
    try:
        print(
            f"HITS failed qualification\t{total_cnt - passed_cnt}\tavg time spent\t{(total_time_completed_in_min - passed_time) / (total_cnt - passed_cnt)} mins"
        )
    except:
        pass


# %%
def retrieve_units(run_id: int) -> list:
    db = LocalMephistoDB()
    units = db.find_units(task_run_id=run_id)
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_units.append(unit)
    return completed_units


# %%
def increment_dict(dict: dict, key: str) -> dict:
    temp_dict = dict
    if key not in temp_dict:
        temp_dict[key] = 1
    else:
        temp_dict[key] += 1
    return temp_dict


# %%
def plot_OS_browser(run_id: int) -> None:
    completed_units = retrieve_units(run_id)
    db = LocalMephistoDB()
    data_browser = DataBrowser(db=db)
    browsers = {}
    browser_versions = {}
    OSs = {}
    mobile = {"yes": 0, "no": 0}
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        user_agent = json.loads(data["data"]["outputs"]["userAgent"])
        browser = user_agent["browser"]["name"]
        browsers = increment_dict(browsers, browser)
        browser_version = browser + str(user_agent["browser"]["v"])
        browser_versions = increment_dict(browser_versions, browser_version)
        OSs = increment_dict(OSs, user_agent["browser"]["os"])
        if user_agent["mobile"]:
            mobile["yes"] += 1
        else:
            mobile["no"] += 1

    plot_hist(browsers, xlabel="Browsers", ylabel=None)
    plot_hist(browser_versions, xlabel="Browser Versions", ylabel=None)
    plot_hist(OSs, xlabel="OS's", ylabel=None)
    plot_hist(mobile, xlabel="On Mobile", ylabel=None)
    return


# %%
def timing_charts(run_id: int) -> None:
    completed_units = retrieve_units(run_id)
    db = LocalMephistoDB()
    data_browser = DataBrowser(db=db)
    workers = {"total": []}
    unit_timing = {"total": [], "end": []}
    question_results = {1: [], 2: [], 3: [], 4: []}
    pass_rates = {1: [], 2: [], 3: [], 4: []}
    starttime = math.inf
    endtime = -math.inf
    feedback = []
    num_correct_hist = []
    bug_count = 0
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        worker = Worker.get(db, data["worker_id"]).worker_name
        workers["total"].append(worker)
        starttime, endtime, unit_timing = hit_timing(data["data"], starttime, endtime, unit_timing)

        outputs = data["data"]["outputs"]
        feedback.append(outputs["feedback"])
        if outputs["bug"] == "true":
            bug_count += 1
        num_correct = 0
        for q in question_results.keys():
            key = "q" + str(q) + "Answer"
            question_results[q].append(outputs[key])
            if outputs[key] == "true":
                num_correct += 1
        num_correct_hist.append(num_correct)

    print(f"Job start time: {datetime.fromtimestamp(starttime)}")
    print(f"Job end time: {datetime.fromtimestamp(endtime)}")

    plot_hist_sorted(
        unit_timing["total"], cutoff=1200, target_val=600, xlabel="", ylabel="Total HIT Time (sec)"
    )
    calc_percentiles(unit_timing["total"], "HIT Length")

    for q in question_results.keys():
        results_dict = Counter(question_results[q])
        pass_rates[q] = (
            results_dict["true"] / (results_dict["true"] + results_dict["false"])
        ) * 100
        print(
            f"Question #{q} pass rate: {(results_dict['true']/(results_dict['true'] + results_dict['false']))*100:.1f}%"
        )
    plot_hist(pass_rates, xlabel="Question #", ylabel=f"Pass Rate %")
    print(
        f"Number of workers who didn't get any right: {len([x for x in num_correct_hist if x == 0])}"
    )

    keys = range(len(num_correct_hist))
    vals_dict = dict(zip(keys, num_correct_hist))
    plot_hist(vals_dict, xlabel="HIT #", ylabel="# Correct", ymax=4)

    print(f"Number of workers who experienced a window crash: {bug_count}")
    print(feedback)


# %%
def hit_timing(
    content: dict, starttime: int, endtime: int, unit_timing: dict
) -> Tuple[int, int, dict]:
    HIT_start_time = content["times"]["task_start"]
    HIT_end_time = content["times"]["task_end"]
    unit_timing["total"].append(HIT_end_time - HIT_start_time)
    unit_timing["end"].append(HIT_end_time)
    if HIT_start_time < starttime:
        starttime = HIT_start_time
    if HIT_start_time > endtime:
        endtime = HIT_end_time
    return starttime, endtime, unit_timing


# %%
def calc_percentiles(data: list, label: str) -> None:
    real_data = [x for x in data if x > 0]
    tenth = np.percentile(real_data, 10)
    median = np.median(real_data)
    nintieth = np.percentile(real_data, 90)
    print(f"{label} tenth percentile: {tenth:.1f}")
    print(f"{label} median {median:.1f}")
    print(f"{label} nintieth percentile: {nintieth:.1f}")


# %%
def plot_hist(
    dictionary: dict,
    ylabel: str,
    target_val: float = None,
    xlabel: str = "Turker Id",
    ymax: float = None,
) -> None:
    plt.bar(list(dictionary.keys()), dictionary.values(), color="g")
    if target_val:
        line = [target_val] * len(dictionary)
        plt.plot(dictionary.keys(), line, color="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()


# %%
def plot_hist_sorted(
    values: list,
    ylabel: str,
    cutoff: float = None,
    target_val: float = None,
    xlabel: str = None,
    ymax: float = None,
) -> None:
    if cutoff:
        values = [cutoff if x > cutoff else x for x in values]
    values.sort()
    keys = range(len(values))
    vals_dict = dict(zip(keys, values))
    plot_hist(vals_dict, target_val=target_val, xlabel=xlabel, ylabel=ylabel, ymax=ymax)


# %%
def plot_scatter(
    xs: list,
    ys: list,
    ylabel: str,
    s: list = None,
    target_val: float = None,
    xlabel: str = "Turker Id",
    ymax: float = None,
) -> None:
    plt.scatter(xs, ys, s, color="g")
    if target_val:
        line = [target_val] * len(xs)
        plt.plot(xs, line, color="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()


# %%
