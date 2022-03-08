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
import pandas as pd
import re
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
    qual_name = "PILOT_ALLOWLIST_QUAL_0920_0"
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
def timing_charts(run_ids: list) -> None:
    combined_usability = []
    combined_command_num = []
    combined_stoplight = []
    combined_parse_errs = []
    for run_id in run_ids:
        completed_units = retrieve_units(run_id)
        db = LocalMephistoDB()
        data_browser = DataBrowser(db=db)
        workers = []
        usability = []
        self_rating = []
        command_num = []
        command_list = []
        starttime = math.inf
        endtime = -math.inf
        HITtime = []
        unit_num = 1
        parsing_errors = []
        parsing_error_counts = []
        stoplight_scores = []
        for unit in completed_units:
            data = data_browser.get_data_from_unit(unit)
            worker = Worker(db, data["worker_id"]).worker_name
            workers.append(worker)
            content = data["data"]
            HIT_start_time = content["times"]["task_start"]
            HIT_end_time = content["times"]["task_end"]
            HITtime.append(HIT_end_time - HIT_start_time)
            if HIT_start_time < starttime:
                starttime = HIT_start_time
            if HIT_start_time > endtime:
                endtime = HIT_end_time
            outputs = content["outputs"]
            try:
                usability.append(int(outputs["usability-rating"]))
            except:
                usability.append(0)
            try:
                self_rating.append(int(outputs["self-rating"]))
            except:
                self_rating.append(0)
            clean_click_string = outputs["clickedElements"].replace("'", "")
            clicks = json.loads(clean_click_string)
            command_count = 0
            parsing_error_count = 0
            if clicks:
                for click in clicks:

                    # Count and collect commands
                    if "command" in click["id"]:
                        command_list.append(click["id"]["command"].split("|")[0])
                        command_count += 1

                    # Count and collect parsing errors
                    if "parsing_error" in click["id"]:
                        if click["id"]["parsing_error"]:
                            parsing_error_count += 1
                            parsing_errors.append(
                                click["id"]["msg"] + "|" + json.dumps(click["id"]["action_dict"])
                            )

                    # Collect stoplight scores
                    if "interactionScores" in click["id"]:
                        if click["id"]["interactionScores"]["stoplight"]:
                            stoplight_scores.append(click["id"]["interactionScores"]["stoplight"])

            # Set logs to 0 for logs that don't exist
            if len(stoplight_scores) < unit_num:
                stoplight_scores.append(0)

            # End of unit bookkeeping
            command_num.append(command_count)
            parsing_error_counts.append(parsing_error_count)
            unit_num += 1

        combined_command_num += command_num
        combined_parse_errs += parsing_error_counts
        combined_usability += usability
        combined_stoplight += stoplight_scores

    actual_usability = [i for i in combined_usability if i > 0]
    print(f"Average usability %: {((sum(actual_usability)*100)/(7*len(actual_usability))):.1f}")
    print(f"Average stoplight score: {(sum(combined_stoplight)/len(combined_stoplight)):.1f}")

    # combined_usability.sort()
    # keys = range(len(combined_usability))
    # u_dict = dict(zip(keys, combined_usability))
    # plot_hist(u_dict, xlabel="", ylabel="Usability Score", ymax=7)
    combined_command_num.sort()
    keys = range(len(combined_command_num))
    c_dict = dict(zip(keys, combined_command_num))
    plot_hist(
        c_dict, target_val=5, xlabel="HIT Index", ylabel="Number of commands per HIT", ymax=20
    )
    # combined_parse_errs.sort()
    # keys = range(len(combined_parse_errs))
    # error_dict = dict(zip(keys, combined_parse_errs))
    # plot_hist(error_dict, xlabel="", ylabel="Number of parsing errors labeled per HIT")
    combined_stoplight.sort()
    keys = range(len(combined_stoplight))
    score_dict = dict(zip(keys, combined_stoplight))
    plot_hist(
        score_dict, target_val=6.5, xlabel="HIT Index", ylabel='HIT Performance "Stoplight" Score'
    )


#%%
def save_commands_to_file(commands, parsing_errors, task_errors, turk_output_directory):
    os.chdir(turk_output_directory)
    with open("mephisto_commands.txt", "w") as comm_file:
        comm_file.write(str(commands))
    with open("mephisto_parsing_errors.txt", "w") as err_file:
        err_file.write(str(parsing_errors))
    with open("mephisto_task_errors.txt", "w") as err_file:
        err_file.write(str(task_errors))
    return


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
        "{turk_logs_dir}/**/nsp_outputs.csv".format(turk_logs_dir=turk_output_directory)
    )
    # Only S3 logs that successfully recorded interaction data have a metadata file
    logs_with_no_commands = [
        csv_path
        for csv_path in nsp_output_folders
        if not os.path.exists(os.path.join(os.path.dirname(csv_path), meta_fname))
    ]

    return [os.path.basename(os.path.dirname(path)) for path in logs_with_no_commands]


#%%
def compare_worker_ids(total, read_instructions, timer_off, submit_command, logs):
    all_lists = [total, read_instructions, timer_off, submit_command, logs]
    list_names = [
        "'Full List'",
        "'Read Instructions'",
        "'Turned Timer Off'",
        "'Submitted a command'",
        "'Recorded S3 Logs'",
    ]
    print(f"Total number of units: {len(total)}")
    print(f"Worker HIT count: {Counter(total)}")
    print(f"Number of unique workers: {len(np.unique(total))}")
    print(f"Median # HITs per worker: {np.median(list(Counter(total).values()))}")
    print(f"Number of units who read instructions: {len(read_instructions)}")
    print(f"Number of units who turned the timer off: {len(timer_off)}")
    print(f"Number of units who submitted a command (recorded by Mephisto): {len(submit_command)}")
    print(f"Number of units who submitted logs to S3: {len(logs)}")
    print(
        f"Number of workers who submitted a job but did not read instructions: {len([x for x in total if x not in read_instructions])}"
    )
    print(
        f"Number of workers who submitted a job but did not turn the timer off: {len([x for x in total if x not in timer_off])}"
    )
    print(
        f"Number of workers who submitted a job but did not submit a command: {len([x for x in total if x not in submit_command])}"
    )
    print(
        f"Number of workers who submitted a job but did not log to S3: {len([x for x in total if x not in logs])}"
    )
    print(
        f"Number of workers who read instructions but did not turn the timer off: {len([x for x in read_instructions if x not in timer_off])}"
    )
    print(
        f"Number of workers who read instructions but did not submit a command: {len([x for x in read_instructions if x not in submit_command])}"
    )
    print(
        f"Number of workers who turned the timer off but did not submit a command: {len([x for x in timer_off if x not in submit_command])}"
    )
    print(
        f"Number of workers who turned the timer off but did not log to S3: {len([x for x in timer_off if x not in logs])}"
    )
    print(
        f"Number of workers who submitted a command but did not log to S3: {len([x for x in submit_command if x not in logs])}"
    )
    print(
        f"Number of workers who submitted a log to S3 but did not submit a command: {len([x for x in logs if x not in submit_command])}"
    )

    for i, l in enumerate(all_lists):
        d1 = Counter(l)
        for j, k in enumerate(all_lists):
            d2 = Counter(k)
            for key in d1.keys():
                try:
                    if d1[key] > d2[key]:
                        print(
                            f"{key} appears in {list_names[i]} {d1[key] - d2[key]} times more than in {list_names[j]}"
                        )
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
                if meta["turk_worker_id"] == turk_id:
                    readfile = True
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
    AC = [
        "build",
        "move",
        "destroy",
        "dance",
        "get",
        "tag",
        "dig",
        "copy",
        "undo",
        "fill",
        "spawn",
        "answer",
        "stop",
        "resume",
        "come",
        "go",
    ]
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

    print(f"num_ori {len_ori}")
    print(f"num_dedup {len_dedup}")
    print(f"dup_rate {((len_ori - len_dedup) / len_ori * 100):.1f}%")
    print(f"avg_len {avg_len}")
    print(f"valid {interested}")
    print(f"valid rate {(interested / len_ori * 100):.1f}%")


#%%
def plot_hist(dictionary, ylabel, target_val=None, xlabel="Turker Id", ymax=None):
    plt.bar(list(dictionary.keys()), dictionary.values(), color="g")
    if target_val:
        line = [target_val] * len(dictionary)
        plt.plot(dictionary.keys(), line, color="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()


#%%
def plot_scatter(xs, ys, ylabel, s=None, target_val=None, xlabel="Turker Id", ymax=None):
    plt.scatter(xs, ys, s, color="g")
    if target_val:
        line = [target_val] * len(xs)
        plt.plot(xs, line, color="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()


#%%
def plot_dual_hist(
    x, y1, y2, ylabel_1="Num of commands with 0 execution time", ylabel_2="Num of HITs completed"
):
    x_locs = np.arange(len(x))
    plt.bar(x_locs, y1, 0.3, color="g")
    plt.bar(x_locs + 0.3, y2, 0.3, color="r")
    plt.xticks(x_locs, x, rotation="vertical")

    pop_a = mpatches.Patch(color="g", label=ylabel_1)
    pop_b = mpatches.Patch(color="r", label=ylabel_2)
    plt.legend(handles=[pop_a, pop_b])
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
            cmd_cnt = len(csv_file["command"])
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

    if filename == "nsp_outputs":
        plot_hist(
            time_dist,
            xlabel="Time Between Comamnds (sec)",
            ylabel="Count of Time Between Commands",
        )
        plot_hist(len_dist, xlabel="Command Length (words)", ylabel="Command Length Count")
        # plot_hist(time_cmd_cnt_map, ylabel="Command Count per Command Time Length")
        # plot_hist(time_turker_map, ylabel="Command Time Length per Turker")
        # plot_hist(cmd_cnt_turker_map, ylabel="Command Count per Turker")

    if all_turk_interactions is None:
        return []

    # Drop duplicates
    all_turk_interactions.drop_duplicates()
    print(f"Stats from S3 command logs:")
    get_stats(list(all_turk_interactions["command"]))
    # return all commands as a list
    return list(set(all_turk_interactions["command"]))
