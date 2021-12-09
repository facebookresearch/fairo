#%%
"""
Get stats and plot for commands

Use it in 3 steps:

1. Pull log files from S3 to a local dir D1
2. Use read_s3_bucket to unzip all log tarballs from D1 to a new dir D2
3. Use read_turk_logs to extract all commands from D2 and do some plotting
"""
#%%
import glob
from numpy import Inf, Infinity
import pandas as pd
import re
from datetime import datetime
import argparse
import os
import tarfile
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from mephisto.data_model.worker import Worker

pd.set_option("display.max_rows", 10)

#%%
def check_run_status(run_id: int) -> None:
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
            completed_units .append(unit)
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
    qual_name = "PILOT_ALLOWLIST_QUAL_0920_0"
    turkers_with_mturk_qual_cnt = 0
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        duration = (data["data"]["times"]["task_end"] - data["data"]["times"]["task_start"]) / 60 # in minutes
        if (duration > 0):
            total_time_completed_in_min += duration
            total_cnt += 1
        worker_name = db.get_worker(worker_id=unit.worker_id)["worker_name"]
        turkers_with_mturk_qual_cnt += 1
        worker = db.find_workers(worker_name=worker_name)[0]
        if worker.get_granted_qualification(qual_name):
            passed_time += duration
            passed_cnt += 1

    print(f"For mephisto/mturk debug: total num: {total_cnt}, # who pass mturk qual: {turkers_with_mturk_qual_cnt}")
    print(f"Total completed HITS\t\t{total_cnt}\tavg time spent\t{total_time_completed_in_min / total_cnt} mins")
    print(f"HITS passed qualification\t{passed_cnt}\tavg time spent\t{passed_time / passed_cnt} mins")
    try:
        print(f"HITS failed qualification\t{total_cnt - passed_cnt}\tavg time spent\t{(total_time_completed_in_min - passed_time) / (total_cnt - passed_cnt)} mins")
    except:
        pass

#%%
def retrieve_units(run_id: int) -> list:
    db = LocalMephistoDB()
    units = db.find_units(task_run_id=run_id)
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_units.append(unit)
    return completed_units

#%%
def increment_dict(dict, key):
    temp_dict = dict
    if key not in temp_dict:
        temp_dict[key] = 1
    else:
        temp_dict[key] += 1
    return temp_dict

#%%
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

#%%
def timing_charts(run_id: int, s3_bucket: int) -> None:
    completed_units = retrieve_units(run_id)
    db=LocalMephistoDB()
    data_browser = DataBrowser(db=db)
    workers = {"total": [], "read_instructions": [], "timer_on": [], "timer_off": [], "sent_command": []}
    ratings = {"usability": [], "self": []}
    inst_timing = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }
    unit_timing = {"total": [],"read": [], "pre_interact": [], "interact": [], "pre_first_command": [], "post_last_command": [], "rating": [], "end": []}
    command_count_lists = {"total": [], "parsing_error": [], "task_error": [], "no_error": []}
    command_processing_times  = { 'send': [], 'NSP': [], 'plan': [], 'execute': [], 'total': [] }
    command_timing_by_hit = { 'start': [], 'end': [] }
    command_lists = {"total": [], "timeout": [], "singleton": [], "parsing_errors": [], "task_errors": []}
    scores = {"creativity": [], "diversity": [], "stoplight": []}
    starttime = math.inf
    endtime = -math.inf
    unit_num = 1
    no_error_dict_list = []
    bonus_list = []
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        worker = Worker(db, data["worker_id"]).worker_name
        workers["total"].append(worker)
        starttime, endtime, unit_timing = hit_timing(data["data"], starttime, endtime, unit_timing)
        
        outputs = data["data"]["outputs"]
        if outputs["usability-rating"]: ratings["usability"].append(int(outputs["usability-rating"]))
        if outputs["self-rating"]: ratings["self"].append(int(outputs["self-rating"]))

        clean_click_string = outputs["clickedElements"].replace("'", "")
        clicks = json.loads(clean_click_string)
        command_counts = {"total": 0, "parsing_error": 0, "task_error": 0, "no_error": 0}
        benchmarks = {"last_status": 0, "last_pg_read": 0, "interaction_start": 0, "interaction_end": None, "command_start": 0, "first_command": True}
        command_timing = {"start": [], "end": []}
        prev_page = 1
        for click in clicks:
            # Instruction timing metrics
            if 'start' in click["id"] or 'page' in click["id"] or 'instructions' in click["id"]:
                inst_timing, benchmarks, prev_page, unit_timing, workers = instruction_timing(click, inst_timing, benchmarks, prev_page, unit_timing, workers, worker)
            # Interaction timing metrics
            if click["id"] == 'timerON' or click["id"] == 'timerOFF':
                benchmarks, unit_timing, workers = interaction_timing(click, benchmarks, unit_timing, workers, worker)
            # Count and collect commands and errors
            if "command" in click["id"] or "parsing_error" in click["id"]:
                command_lists, command_counts, no_error_dict_list = commands_and_errors(click["id"], command_lists, command_counts, no_error_dict_list)
            # Collect stoplight scores
            if "interactionScores" in click["id"]:
                scores, bonus_list = interaction_scores(click["id"]["interactionScores"], scores, bonus_list, worker)
            # Command timing metrics:
            command_processing_msgs = ['goToAgentThinking', 'received', 'done_thinking', 'executing', 'goToMessage']
            if click["id"] in command_processing_msgs:
                benchmarks, unit_timing, command_timing, command_processing_times, command_lists, workers = command_timing_metrics(click, benchmarks, unit_timing, command_timing, command_processing_times, command_lists, workers, worker)

        # Record end-of-unit timing meterics
        if benchmarks["interaction_end"]:
            unit_timing["rating"].append(unit_timing["end"][-1] - round((benchmarks["interaction_end"]/1000)))
            if benchmarks["last_status"]:
                unit_timing["post_last_command"].append((benchmarks["interaction_end"] - benchmarks["last_status"])/1000)
        unit_timing["pre_first_command"] = [x for x in unit_timing["pre_first_command"] if x<1000000]

        # Set logs to 0 for logs that don't exist
        ratings = match_length(ratings, unit_num)
        inst_timing = match_length(inst_timing, unit_num)
        unit_timing = match_length(unit_timing, unit_num)
        scores = match_length(scores, unit_num)

        # End of unit bookkeeping
        for key in command_count_lists.keys():
            command_count_lists[key].append(command_counts[key])
        for key in command_timing_by_hit.keys():
            command_timing_by_hit[key].append(command_timing[key])
        if command_counts["total"] == 1:
            command_lists["singleton"].append(command_lists["total"][-1])
        unit_num += 1

        ### END OF JOB ANALYTICS LOOP ###

    parsed_path = os.path.join("/private/home/ethancarlson/.hitl/parsed/", str(s3_bucket))
    
    # Create a file containing the bonus payouts for turkers who receive performance incentives
    create_bonus_payment_file(s3_bucket, bonus_list)

    #workers_logs = retrieve_turker_ids(parsed_path, "nsp_outputs")
    #compare_worker_ids(workers, workers_read_instructions, workers_timer_off, workers_sent_command, workers_logs)
    #get_commands_from_turk_id('A4D99Y82KOLC8', parsed_path)
    
    # See the S3 logs that don't contain commands:
    # print(logs_with_no_commands(parsed_path))

    #save_commands_to_file(command_lists["total"], command_lists["parsing_errors"], command_lists["task_errors"], parsed_path)

    print_stats_from_mephisto(unit_num, starttime, endtime, command_lists["total"], command_count_lists, unit_timing["total"], ratings["usability"], ratings["self"], scores)
    
    #Compare Mephisto and S3 error lists for discrepanicies
    #compare_s3_and_meph_errors(parsed_path, command_lists, no_error_dict_list)

    # Report problematic commands
    # print(f"Commands from HITs that only issued one command: {singleton_commands}")
    # print(f"Commands that triggered the timeout: {command_lists['timeout']}")

    # See the median, 10th and 90th percentile HIT timing anatomy
    #hit_timing_anatomy(unit_timing, inst_timing, command_processing_times, command_timing_by_hit, unit_num)
    
    produce_plots(ratings, unit_timing, command_count_lists, scores, inst_timing, command_processing_times)

#%%
def compare_s3_and_meph_errors(parsed_path, command_lists, no_error_dict_list):
    S3_errors = read_turk_logs(parsed_path, "error_details")
    meph_errors = [x.split('|')[0] for x in command_lists["parsing_errors"]] + [x.split('|')[0] for x in command_lists["task_errors"]]
    S3_not_meph = [x for x in S3_errors if x not in meph_errors]
    meph_not_S3 = [x for x in meph_errors if x not in S3_errors]
    print(f"Number of errors in S3 but not Mephisto: {len(S3_not_meph)}")
    print(f"Errors in S3 but not Mephisto: {S3_not_meph}")
    print(f"Errors in Mephisto but not S3: {meph_not_S3}")
    maybe_bad_error_dicts = []
    for err in S3_not_meph:
        for d in no_error_dict_list:
            if d["msg"] == err:
                maybe_bad_error_dicts.append(d)
    print(f"Num S3/Mephisto discrepancies found in no_error list: {len(maybe_bad_error_dicts)}")
    print(f"Maybe bad error dicts: {maybe_bad_error_dicts}")

#%%
def create_bonus_payment_file(s3_bucket, bonus_list):
    bonus_path = os.path.join("/private/home/ethancarlson/.hitl/bonus/", str(s3_bucket))
    os.makedirs(bonus_path, exist_ok=True)
    bonus_file = os.path.join(bonus_path, "performance_bonuses.txt")
    with open(bonus_file, "w+") as f:
        for bonus in bonus_list:
            f.write(bonus)

#%%
def command_timing_metrics(click, benchmarks, unit_timing, command_timing, command_processing_times, command_lists, workers, worker):
    if click["id"] == 'goToAgentThinking':
        command_start_time = click["timestamp"]
        command_timing["start"].append(command_start_time)
        benchmarks["last_status"] = click["timestamp"]
        if benchmarks["first_command"]:
            workers["sent_command"].append(worker)
            unit_timing["pre_first_command"].append(round((command_start_time - benchmarks["interaction_start"])/1000))
            benchmarks["first_command"] = False
    if click["id"] == 'received':
        command_processing_times['send'].append((click["timestamp"] - benchmarks["last_status"])/1000)
        benchmarks["last_status"] = click["timestamp"]
    if click["id"] == 'done_thinking':
        command_processing_times['NSP'].append((click["timestamp"] - benchmarks["last_status"])/1000)
        benchmarks["last_status"] = click["timestamp"]
    if click["id"] == 'executing':
        command_processing_times['plan'].append((click["timestamp"] - benchmarks["last_status"])/1000)
        benchmarks["last_status"] = click["timestamp"]
    if click["id"] == 'goToMessage':
        command_processing_times['execute'].append((click["timestamp"] - benchmarks["last_status"])/1000)
        command_processing_times['total'].append((click["timestamp"] - command_start_time)/1000)
        # if the command took a long time, remember it
        if command_processing_times['total'][-1] > 49:
            command_lists["timeout"].append(command_lists["total"][-1])
        command_timing["end"].append(click["timestamp"])
        # Reset and set up for next command:
        benchmarks["last_status"] = click["timestamp"]
        # Append 0s to any steps that were skipped
        num_commands = max([len(value) for value in command_processing_times.values()])
        for status in command_processing_times.keys():
            if len(command_processing_times[status]) < num_commands:
                command_processing_times[status].append(0)

    return benchmarks, unit_timing, command_timing, command_processing_times, command_lists, workers

#%%
def produce_plots(ratings, unit_timing, command_count_lists, scores, inst_timing, command_processing_times):
    #Turker ratings plots
    plot_hist_sorted(ratings["usability"], xlabel="", ylabel="Usability Score", ymax=7)
    plot_hist_sorted(ratings["self"], xlabel="", ylabel="Self Rated Performance Score", ymax=5)

    # Command count and error plots
    plot_hist_sorted(command_count_lists['total'], target_val=5, xlabel="", ylabel="Number of commands per HIT")
    plot_hist_sorted(command_count_lists['parsing_error'], xlabel="", ylabel="Number of parsing errors labeled per HIT")
    plot_hist_sorted(command_count_lists['task_error'], xlabel="", ylabel="Number of task errors labeled per HIT")

    # HIT timing breakdown plots
    plot_hist_sorted(unit_timing["total"], cutoff=900, target_val=480, xlabel="", ylabel="Total HIT Time (sec)")
    plot_hist_sorted(unit_timing["read"], cutoff=360, target_val=180, xlabel="", ylabel="Instructions Read Time (sec)")
    plot_hist_sorted(unit_timing["pre_interact"], cutoff=100, target_val=30, xlabel="", ylabel="Time between instructions and interaction start (sec)")
    plot_hist_sorted(unit_timing["interact"], cutoff=750, target_val=300, xlabel="", ylabel="Interaction time (sec)")
    
    # Interaction quality scores
    plot_hist_sorted(scores["creativity"], xlabel="", ylabel="Creativity Scores")
    plot_hist_sorted(scores["diversity"], xlabel="", ylabel="Diversity Scores")
    plot_hist_sorted(scores["stoplight"], target_val=6.5, xlabel="", ylabel="Stoplight Scores")

    # Timing breakdowns for each instruction page
    # plot_instruction_page_timing(inst_timing)

    # Timing breakdowns for each stage of command processing
    plot_command_stage_timing(command_processing_times)

    # Comparison scatter plots
    # plot_scatter(xs=command_count_lists['total'], ys=unit_timing["total"], xlabel="# of Commands", ylabel="HIT Length")
    # plot_data_count = Counter(zip(command_count_lists['total'], ratings["usability"]))
    # bubble_size = [plot_data_count[(command_count_lists['total'][i],ratings["usability"][i])]*30 for i in range(len(command_count_lists['total']))]
    # plot_scatter(xs=command_count_lists['total'], ys=ratings["usability"], s=bubble_size, xlabel="# of Commands", ylabel="Usability Score")
    # plot_data_count = Counter(zip(command_count_lists['total'], ratings["self"]))
    # bubble_size = [plot_data_count[(command_count_lists['total'][i],ratings["self"][i])]*30 for i in range(len(command_count_lists['total']))]
    # plot_scatter(xs=command_count_lists['total'], ys=ratings["self"], s=bubble_size, xlabel="# of Commands", ylabel="Self Rating")
    # plot_scatter(xs=command_count_lists['total'], ys=scores["stoplight"], xlabel="# of Commands", ylabel="Stoplight Score")

#%%
def interaction_timing(click, benchmarks, unit_timing, workers, worker):
    if click["id"] == 'timerON':
        benchmarks["interaction_start"] = click["timestamp"]
        unit_timing["pre_interact"].append(round((benchmarks["interaction_start"] - benchmarks["last_pg_read"])/1000))
        workers["timer_on"].append(worker)
    if click["id"] == 'timerOFF':
        benchmarks["interaction_end"] = click["timestamp"]
        unit_timing["interact"].append(round((benchmarks["interaction_end"] - benchmarks["interaction_start"])/1000))
        workers["timer_off"].append(worker)
    return benchmarks, unit_timing, workers

#%%
def hit_timing_anatomy(unit_timing, inst_timing, command_processing_times, command_timing_by_hit, unit_num):

    command_times_by_order = { 1: [], 2: [], 3: [], 4: [], 5: [] }
    for j in range(5):
        for i in range(unit_num-1):
            try:
                start = command_timing_by_hit["start"][i][j]
                end = command_timing_by_hit["end"][i][j]
                command_times_by_order[(j+1)].append((end - start)/1000)
            except:
                pass

    calc_percentiles(unit_timing["read"], "Total Read Time")
    for page in inst_timing.keys():
        if page == 0: continue
        calc_percentiles(inst_timing[page], f"Page #{page} Read Time")
    calc_percentiles(unit_timing["pre_interact"], "Pre-Interaction Time")
    calc_percentiles(unit_timing["interact"], "Interaction Time")
    calc_percentiles(unit_timing["pre_first_command"], "Time After Start Before First Command")
    calc_percentiles(unit_timing["post_last_command"], "Time After Last Command Before End")
    calc_percentiles(unit_timing["rating"], "Time After Interaction End")
    calc_percentiles(command_processing_times['total'], "Avg Command Time")
    for order in command_times_by_order.keys():
        calc_percentiles(command_times_by_order[order], f"Command #{order} Time")

#%%
def plot_command_stage_timing(command_processing_times):
    for status in command_processing_times.keys():
        command_processing_times[status] = [0 if x<0 else x for x in command_processing_times[status]]
        command_processing_times[status] = [50 if x>50 else x for x in command_processing_times[status]]
        command_processing_times[status].sort()
        keys = range(len(command_processing_times[status]))
        command_dict = dict(zip(keys, command_processing_times[status]))
        plot_hist(command_dict, xlabel="", ylabel=f"Command {status} time (sec)")

#%%
def plot_instruction_page_timing(inst_timing):
    for page in inst_timing.keys():
        inst_timing[page] = [0 if x<0 else x for x in inst_timing[page]]
        inst_timing[page] = [90 if x>90 else x for x in inst_timing[page]]
        inst_timing[page].sort()
        keys = range(len(inst_timing[page]))
        page_dict = dict(zip(keys, inst_timing[page]))
        plot_hist(page_dict, xlabel="", ylabel=f"Page {page} read time (sec)")

#%%
def commands_and_errors(click_id, command_lists, command_counts, no_error_dict_list):
    if "command" in click_id:
        command_counts["total"] += 1
        command_lists["total"].append(click_id["command"].split('|')[0])

    if "parsing_error" in click_id:
        if click_id["parsing_error"]:
            command_counts["parsing_error"] += 1
            command_lists["parsing_errors"].append(click_id["msg"] + "|" + json.dumps(click_id["action_dict"]))
        elif click_id["task_error"]:
            command_counts["task_error"] += 1
            command_lists["task_errors"].append(click_id["msg"] + "|" + json.dumps(click_id["action_dict"]))
        else:
            command_counts["no_error"] += 1
            no_error_dict_list.append(click_id)
    return command_lists, command_counts, no_error_dict_list
#%%
def hit_timing(content, starttime, endtime, unit_timing):
    HIT_start_time = content["times"]["task_start"]
    HIT_end_time = content["times"]["task_end"]
    unit_timing["total"].append(HIT_end_time - HIT_start_time)
    unit_timing["end"].append(HIT_end_time)
    if (HIT_start_time < starttime):
        starttime = HIT_start_time
    if (HIT_start_time > endtime):
        endtime = HIT_end_time
    return starttime, endtime, unit_timing

#%%
def instruction_timing(click, inst_timing, benchmarks, prev_page, unit_timing, workers, worker):
    # They might not read all pages...
    if click["id"] == 'start':
        inst_timing[0].append(click["timestamp"])
        benchmarks["last_pg_read"] = click["timestamp"]
    try:
        if 'page-' in click["id"]:
            inst_timing[prev_page].append((click["timestamp"] - benchmarks["last_pg_read"])/1000)
            benchmarks["last_pg_read"] = click["timestamp"]
            prev_page = int(click["id"][-1])
    except:
        pass
    if click["id"] == 'instructions-popup-close':
        inst_timing[5].append((click["timestamp"] - benchmarks["last_pg_read"])/1000)
        unit_timing["read"].append(round((click["timestamp"] - inst_timing[0][-1])/1000))
        benchmarks["last_pg_read"] = click["timestamp"]
        workers["read_instructions"].append(worker)

    return inst_timing, benchmarks, prev_page, unit_timing, workers 
#%%
def print_stats_from_mephisto(unit_num, starttime, endtime, command_list, command_count_lists, HITtime, usability, self_rating, scores):
    actual_usability = [i for i in usability if i > 0]
    actual_self_rating = [i for i in self_rating if i > 0]
    print(f"Units logged: {unit_num-1}")
    print(f"Start time: {datetime.fromtimestamp(starttime)}")
    print(f"End time: {datetime.fromtimestamp(endtime)}")
    print(f"Mephisto command list stats:")
    get_stats(command_list)
    print(f"Total Command count: {sum(command_count_lists['total'])}")
    print(f"Avg Number of commands in Mephisto: {sum(command_count_lists['total'])/len(command_count_lists['total']):.1f}")
    print(f"Total parsing error count: {sum(command_count_lists['parsing_error'])}")
    print(f"Total task error count: {sum(command_count_lists['task_error'])}")
    print(f"Total commands with no error, as determined by error dict: {sum(command_count_lists['no_error'])}")
    print(f"Units w/ no commands in Mephisto: {command_count_lists['total'].count(0)}")
    print(f"Average HIT length (mins): {sum(HITtime)/(60*len(HITtime)):.1f}")
    print(f"Average usability %: {((sum(actual_usability)*100)/(7*len(actual_usability))):.1f}")
    print(f"Average self assessment %: {((sum(actual_self_rating)*100)/(5*len(actual_self_rating))):.1f}")
    print(f"Average creativity score: {(sum(scores['creativity'])/len(scores['creativity'])):.1f}")
    print(f"Average diversity score: {(sum(scores['diversity'])/len(scores['diversity'])):.1f}")
    print(f"Average stoplight score: {(sum(scores['stoplight'])/len(scores['stoplight'])):.1f}")

#%%
def match_length(data: dict, length: int):
    for key in data.keys():
            if len(data[key]) < length:
                data[key].append(0)
    return data

#%%
def save_commands_to_file(commands, parsing_errors, task_errors, turk_output_directory):
    os.chdir(turk_output_directory)
    with open("mephisto_commands.txt", "w+") as comm_file:
        comm_file.write(str(commands))
    with open("mephisto_parsing_errors.txt", "w+") as err_file:
        err_file.write(str(parsing_errors))
    with open("mephisto_task_errors.txt", "w+") as err_file:
        err_file.write(str(task_errors))
    return
    
#%%
def interaction_scores(scores, output_dict, bonus_list, worker):
    if scores["creativity"]: output_dict["creativity"].append(scores["creativity"])
    if scores["diversity"]: output_dict["diversity"].append(scores["diversity"])
    if scores["stoplight"]:
        output_dict["stoplight"].append(scores["stoplight"])
        bonus_list.append(str(worker) + " " + f"{(scores['stoplight'] * 0.30):.2f}" + "\n")
    return output_dict, bonus_list
#%%
def retrieve_turker_ids(turk_output_directory, filename, meta_fname="job_metadata.json"):
    workers = []
    for csv_path in glob.glob(
        "{turk_logs_dir}/**/{csv_filename}".format(
            turk_logs_dir=turk_output_directory, csv_filename=filename + ".csv"
        )
    ):     
        meta_path = os.path.join(os.path.dirname(csv_path), meta_fname)
        if os.path.exists(meta_path):
            with open(meta_path, "r+") as f:
                meta = json.load(f)
                workers.append(meta["turk_worker_id"])
        else:
            pass
    return workers

#%%
def logs_with_no_commands(turk_output_directory, meta_fname="job_metadata.json"):
    nsp_output_folders = glob.glob(
        "{turk_logs_dir}/**/nsp_outputs.csv".format(turk_logs_dir=turk_output_directory))
    # Only S3 logs that successfully recorded interaction data have a metadata file
    logs_with_no_commands = [csv_path for csv_path in nsp_output_folders if not os.path.exists(os.path.join(os.path.dirname(csv_path), meta_fname))]
    
    return [os.path.basename(os.path.dirname(path)) for path in logs_with_no_commands]


#%%
def compare_worker_ids(total, read_instructions, timer_off, submit_command, logs):
    all_lists = [total, read_instructions, timer_off, submit_command, logs]
    list_names = ["'Full List'", "'Read Instructions'", "'Turned Timer Off'", "'Submitted a command'", "'Recorded S3 Logs'"]
    print(f"Total number of units: {len(total)}")
    print(f"Worker HIT count: {Counter(total)}")
    print(f"Number of unique workers: {len(np.unique(total))}")
    print(f"Median # HITs per worker: {np.median(list(Counter(total).values()))}")
    print(f"Number of units who read instructions: {len(read_instructions)}")
    print(f"Number of units who turned the timer off: {len(timer_off)}")
    print(f"Number of units who submitted a command (recorded by Mephisto): {len(submit_command)}")
    print(f"Number of units who submitted logs to S3: {len(logs)}")
    print(f"Number of workers who submitted a job but did not read instructions: {len([x for x in total if x not in read_instructions])}")
    print(f"Number of workers who submitted a job but did not turn the timer off: {len([x for x in total if x not in timer_off])}")
    print(f"Number of workers who submitted a job but did not submit a command: {len([x for x in total if x not in submit_command])}")
    print(f"Number of workers who submitted a job but did not log to S3: {len([x for x in total if x not in logs])}")
    print(f"Number of workers who read instructions but did not turn the timer off: {len([x for x in read_instructions if x not in timer_off])}")
    print(f"Number of workers who read instructions but did not submit a command: {len([x for x in read_instructions if x not in submit_command])}")
    print(f"Number of workers who turned the timer off but did not submit a command: {len([x for x in timer_off if x not in submit_command])}")
    print(f"Number of workers who turned the timer off but did not log to S3: {len([x for x in timer_off if x not in logs])}")
    print(f"Number of workers who submitted a command but did not log to S3: {len([x for x in submit_command if x not in logs])}")
    print(f"Number of workers who submitted a log to S3 but did not submit a command: {len([x for x in logs if x not in submit_command])}")

    for i,l in enumerate(all_lists):
        d1 = Counter(l)
        for j,k in enumerate(all_lists):
            d2 = Counter(k)
            for key in d1.keys():
                try:
                    if d1[key] > d2[key]:
                        print(f"{key} appears in {list_names[i]} {d1[key] - d2[key]} times more than in {list_names[j]}")
                except:
                    print(f"{key} appears in {list_names[i]} but not {list_names[j]}")

#%%
def get_commands_from_turk_id(turk_id, turk_output_directory):
    
    commands = []
    for csv_path in glob.glob(
        "{turk_logs_dir}/**/{csv_filename}".format(
            turk_logs_dir=turk_output_directory, csv_filename="nsp_outputs.csv"
        )
    ):
        readfile = False
        meta_path = os.path.join(os.path.dirname(csv_path), "job_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r+") as f:
                meta = json.load(f)
                if meta["turk_worker_id"] == turk_id: readfile = True
        else:
            pass

        if readfile:
            with open(csv_path) as fd:
                # collect the NSP outputs CSV
                csv_file = pd.read_csv(csv_path, delimiter="|")
                commands += list(csv_file["command"])
    
    print(f"Commands from worker {turk_id}: {commands}")
    print(f"Worker {turk_id} command #: {len(commands)}")

#%%
def calc_percentiles(data, label):
    real_data = [x for x in data if x > 0]
    tenth = np.percentile(real_data, 10)
    median = np.median(real_data)
    nintieth = np.percentile(real_data, 90)
    print(f"{label} tenth percentile: {tenth:.1f}")
    print(f"{label} median {median:.1f}")
    print(f"{label} nintieth percentile: {nintieth:.1f}")

#%%
def read_s3_bucket(s3_logs_dir, output_dir):
    print(
        "{s3_logs_dir}/**/{csv_filename}".format(
            s3_logs_dir=s3_logs_dir, csv_filename="logs.tar.gz"
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    # NOTE: This assumes the local directory is synced with the same name as the S3 directory
    pattern = re.compile(r".*turk_logs/(.*)/logs.tar.gz")
    # NOTE: this is hard coded to search 2 levels deep because of how our logs are structured
    for csv_path in glob.glob(
        "{s3_logs_dir}/**/{csv_filename}".format(
            s3_logs_dir=s3_logs_dir, csv_filename="logs.tar.gz"
        )
    ):
        tf = tarfile.open(csv_path)
        timestamp = pattern.match(csv_path).group(1)
        tf.extractall(path="{}/{}/".format(output_dir, timestamp))

#%%
def get_stats(command_list):
    """
    Print some stats of a list of commands
    """
    AC = ["build", "move", "destroy", "dance", "get", "tag", "dig", "copy", "undo", "fill", "spawn", "answer", "stop", "resume", "come", "go"]
    command_list = [c.lower() for c in command_list]
    len_ori = len(command_list)
    l = list(set(command_list))
    len_dedup = len(l)

    total_len = 0
    interested = 0
    for c in l:
        # print(c)
        if any(word in c.lower() for word in AC):
            interested += 1
        total_len += len(c.split())
    avg_len = total_len / len_dedup

    print(f'num_ori {len_ori}')
    print(f'num_dedup {len_dedup}')
    print(f'dup_rate {((len_ori - len_dedup) / len_ori * 100):.1f}%')
    print(f'avg_len {avg_len}')
    print(f'valid {interested}')
    print(f'valid rate {(interested / len_ori * 100):.1f}%')


#%%
def plot_hist(dictionary, ylabel, target_val=None, xlabel="Turker Id", ymax=None):
    plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
    if target_val:
        line = [target_val] * len(dictionary)
        plt.plot(dictionary.keys(), line, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()

#%%
def plot_hist_sorted(values, ylabel, cutoff=None, target_val=None, xlabel=None, ymax=None):
    if cutoff: values = [cutoff if x>cutoff else x for x in values]
    values.sort()
    keys = range(len(values))
    vals_dict = dict(zip(keys, values))
    plot_hist(vals_dict, target_val=target_val, xlabel=xlabel, ylabel=ylabel, ymax=ymax)

#%%
def plot_scatter(xs, ys, ylabel, s=None, target_val=None, xlabel="Turker Id", ymax=None):
    plt.scatter(xs, ys, s, color='g')
    if target_val:
        line = [target_val] * len(xs)
        plt.plot(xs, line, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()

#%%
def plot_dual_hist(x, y1, y2, ylabel_1="Num of commands with 0 execution time", ylabel_2="Num of HITs completed"):
    x_locs = np.arange(len(x))
    plt.bar(x_locs, y1, 0.3, color='g')
    plt.bar(x_locs + 0.3, y2 , 0.3, color='r')
    plt.xticks(x_locs, x, rotation='vertical')

    pop_a = mpatches.Patch(color='g', label=ylabel_1)
    pop_b = mpatches.Patch(color='r', label=ylabel_2)
    plt.legend(handles=[pop_a,pop_b])
    plt.show()

#%%
def read_turk_logs(turk_output_directory, filename, meta_fname="job_metadata.json"):
    # Crawl turk logs directory
    all_turk_interactions = None
    len_dist = {}
    len_dist[15] = 0
    time_dist = {}
    time_turker_map = {}
    turker_hit_cnt = {}
    cmd_cnt_turker_map = {}
    time_cmd_cnt_map = {}
    
    for csv_path in glob.glob(
        "{turk_logs_dir}/**/{csv_filename}".format(
            turk_logs_dir=turk_output_directory, csv_filename=filename + ".csv"
        )
    ):     
        meta_path = os.path.join(os.path.dirname(csv_path), meta_fname)
        if os.path.exists(meta_path):
            with open(meta_path, "r+") as f:
                meta = json.load(f)
                turk_id = meta["turk_worker_id"]
        else:
            pass
        
        cnt = turker_hit_cnt.get(turk_id, 0)
        turker_hit_cnt[turk_id] = cnt + 1

        with open(csv_path) as fd:
            # collect the NSP outputs CSV
            csv_file = pd.read_csv(csv_path, delimiter="|")
            csv_file = csv_file.drop_duplicates(subset=["command"])

            # get time distribution stats
            timelist = list(csv_file["time"])
            for i in range(len(timelist) - 1):
                
                if math.isnan(timelist[i + 1] - timelist[i]):
                    time = -1
                else:
                    time = int(timelist[i + 1] - timelist[i])
                if time > 120:
                    time = 120
                if time not in time_dist:
                    time_dist[time] = 0
                time_dist[time] += 1

            # get command turker maps
            num_rows = len(csv_file.index) if not csv_file.empty else 0
            for index, row in csv_file.iterrows():
                if index >= num_rows - 1:
                    break
                if math.isnan(timelist[index + 1] - timelist[index]):
                    time = -1
                else:
                    time = int(timelist[index + 1] - timelist[index])
                turker_map = time_turker_map.get(time, {})
                cnt = turker_map.get(turk_id, 0)
                turker_map[turk_id] = cnt + 1
                time_turker_map[time] = turker_map

                cmd = row["command"].lower().strip()
                cmd_cnt_map = time_cmd_cnt_map.get(time, {})
                cnt = cmd_cnt_map.get(cmd, 0)
                cmd_cnt_map[cmd] = cnt + 1
                time_cmd_cnt_map[time] = cmd_cnt_map
            
            # get command count stats
            cmd_cnt = len(csv_file['command'])
            if cmd_cnt not in len_dist:
                if cmd_cnt < 15:
                    len_dist[cmd_cnt] = 0
            if cmd_cnt > 15:
                len_dist[15] += 1
            else:
                len_dist[cmd_cnt] += 1

            # get cmd cnt turker map
            turker_map = cmd_cnt_turker_map.get(cmd_cnt, {})
            freq = turker_map.get(turk_id, 0)
            turker_map[turk_id] = freq + 1
            cmd_cnt_turker_map[cmd_cnt] = turker_map

            if all_turk_interactions is None:
                all_turk_interactions = csv_file
            else:
                all_turk_interactions = pd.concat(
                    [all_turk_interactions, csv_file], ignore_index=True
                )

    if (filename == "nsp_outputs"):
        plot_hist(time_dist, xlabel="Time Between Comamnds (sec)", ylabel="Count of Time Between Commands")
        plot_hist(len_dist, xlabel="Command Length (words)", ylabel="Command Length Count")
        #plot_hist(time_cmd_cnt_map, ylabel="Command Count per Command Time Length")
        #plot_hist(time_turker_map, ylabel="Command Time Length per Turker")
        #plot_hist(cmd_cnt_turker_map, ylabel="Command Count per Turker")

    if all_turk_interactions is None:
        return []

    # Drop duplicates
    all_turk_interactions.drop_duplicates()
    print(f"Stats from S3 command logs:")
    get_stats(list(all_turk_interactions["command"]))
    # return all commands as a list
    return list(set(all_turk_interactions["command"]))

#%%
read_s3_bucket("/private/home/ethancarlson/.hitl/20211201131224/turk_logs", "/private/home/ethancarlson/.hitl/parsed/20211201131224")
print("\nNSP Outputs: ")
read_turk_logs("/private/home/ethancarlson/.hitl/parsed/20211201131224", "nsp_outputs")
print("\nError Details: ")
read_turk_logs("/private/home/ethancarlson/.hitl/parsed/20211201131224", "error_details")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # User needs to provide file I/O paths
    parser.add_argument(
        "--s3_bucket", required=True,
        help="where to read s3 logs from eg. '20211025173851'",
    )
    parser.add_argument(
        "--run_id", required=True,
        help="The Mephisto run ID, eg.'218'",
    )
    args = parser.parse_args()

    
