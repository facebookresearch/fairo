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
def timing_charts(run_id: int) -> None:
    db = LocalMephistoDB()
    units = db.find_units(task_run_id=run_id)
    completed_num = 0
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_num += 1
            completed_units .append(unit)

    data_browser = DataBrowser(db=db)
    workers = []
    workers_read_instructions = []
    workers_timer_on = []
    workers_timer_off = []
    workers_sent_command = []
    usability = []
    self_rating = []
    inst_timing = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }
    read_time = []
    pre_interact = []
    interact_time = []
    pre_first_command_time = []
    post_last_command_time = []
    rating_time = []
    command_num = []
    command_timing  = { 'send': [], 'NSP': [], 'plan': [], 'execute': [], 'total': [] }
    command_timing_by_hit = { 'start': [], 'end': [] }
    starttime = math.inf
    endtime = -math.inf
    HITtime = []
    unit_num = 1
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        worker = Worker(db, data["worker_id"]).worker_name
        workers.append(worker)
        content = data["data"]
        HIT_start_time = content["times"]["task_start"]
        HIT_end_time = content["times"]["task_end"]
        if (HIT_start_time < starttime):
            starttime = HIT_start_time
        if (HIT_start_time > endtime):
            endtime = HIT_end_time
        HITtime.append(HIT_end_time - HIT_start_time)
        outputs = content["outputs"]
        try:
            usability.append(int(outputs["usability-rating"]))
        except:
            usability.append(0)
        try:
            self_rating.append(int(outputs["self-rating"]))
        except:
            self_rating.append(0)
        clicks = json.loads(outputs["clickedElements"])
        command_count = 0
        last_status_time = 0
        last_pg_read_time = 0
        interaction_start_time = 0
        interaction_end_time = None
        prev_page = 1
        command_start_times = []
        command_end_times = []
        first_command = True
        for click in clicks:
            # They might not read all of the instructions pages...
            if click["id"] == 'start':
                inst_timing[0].append(click["timestamp"])
                last_pg_read_time = click["timestamp"]
            try:
                if 'page-' in click["id"]:
                    inst_timing[prev_page].append((click["timestamp"] - last_pg_read_time)/1000)
                    last_pg_read_time = click["timestamp"]
                    prev_page = int(click["id"][-1])
            except:
                pass
            if click["id"] == 'instructions-popup-close':
                inst_timing[5].append((click["timestamp"] - last_pg_read_time)/1000)
                read_time.append(round((click["timestamp"] - inst_timing[0][-1])/1000))
                last_pg_read_time = click["timestamp"]
                workers_read_instructions.append(worker)

            # Interaction timing metrics
            if click["id"] == 'timerON':
                interaction_start_time = click["timestamp"]
                pre_interact.append(round((interaction_start_time - last_pg_read_time)/1000))
                workers_timer_on.append(worker)
            if click["id"] == 'timerOFF':
                interaction_end_time = click["timestamp"]
                interact_time.append(round((interaction_end_time - interaction_start_time)/1000))
                workers_timer_off.append(worker)
            
            # Command timing metrics:
            if click["id"] == 'goToAgentThinking':
                command_count += 1
                command_start_time = click["timestamp"]
                command_start_times.append(command_start_time)
                last_status_time = click["timestamp"]
                if first_command:
                    workers_sent_command.append(worker)
                    pre_first_command_time.append(round((command_start_time - interaction_start_time)/1000))
                    first_command = False
            if click["id"] == 'received':
                command_timing['send'].append((click["timestamp"] - last_status_time)/1000)
                last_status_time = click["timestamp"]
            if click["id"] == 'done_thinking':
                command_timing['NSP'].append((click["timestamp"] - last_status_time)/1000)
                last_status_time = click["timestamp"]
            if click["id"] == 'executing':
                command_timing['plan'].append((click["timestamp"] - last_status_time)/1000)
                last_status_time = click["timestamp"]
            if click["id"] == 'goToMessaage':
                command_timing['execute'].append((click["timestamp"] - last_status_time)/1000)
                command_timing['total'].append((click["timestamp"] - command_start_time)/1000)
                command_end_times.append(click["timestamp"])
                # Reset and set up for next command:
                last_status_time = click["timestamp"]
                num_commands = max([len(value) for value in command_timing.values()])
                for status in command_timing.keys():
                    if len(command_timing[status]) < num_commands:
                        command_timing[status].append(0)

        if interaction_end_time:
            rating_time.append(HIT_end_time - round((interaction_end_time/1000)))
            if last_status_time:
                post_last_command_time.append((interaction_end_time - last_status_time)/1000)
        pre_first_command_time = [x for x in pre_first_command_time if x<1000000]

        # Set timing logs to 0 for logs that don't exist
        for page in inst_timing.keys():
            if len(inst_timing[page]) < unit_num:
                inst_timing[page].append(0)
        if len(read_time) < unit_num: read_time.append(0)
        if len(pre_interact) < unit_num: pre_interact.append(0)
        if len(interact_time) < unit_num: interact_time.append(0)

        command_num.append(command_count)
        command_timing_by_hit["start"].append(command_start_times)
        command_timing_by_hit["end"].append(command_end_times)
        unit_num += 1

    command_times_by_order = { 1: [], 2: [], 3: [], 4: [], 5: [] }
    for j in range(5):
        for i in range(unit_num-1):
            try:
                start = command_timing_by_hit["start"][i][j]
                end = command_timing_by_hit["end"][i][j]
                command_times_by_order[(j+1)].append((end - start)/1000)
            except:
                pass
    
    workers_logs = retrieve_turker_ids("/private/home/ethancarlson/.hitl/parsed/20211112115924", "nsp_outputs")
    compare_worker_ids(workers, workers_read_instructions, workers_timer_off, workers_sent_command, workers_logs)

    '''
    promoters = len([i for i in usability if i > 5])
    detractors = len([i for i in usability if i < 5 and i > 0])
    actual_usability = [i for i in usability if i > 0]
    actual_self_rating = [i for i in self_rating if i > 0]

    print(f"Units logged: {unit_num-1}")
    print(f"Start time: {datetime.fromtimestamp(starttime)}")
    print(f"End time: {datetime.fromtimestamp(endtime)}")
    print(f"Avg Number of commands: {sum(command_num)/len(command_num):.1f}")
    print(f"Units w/ no commands: {command_num.count(0)}")
    print(f"Average HIT length (mins): {sum(HITtime)/(60*len(HITtime)):.1f}")
    print(f"Average usability %: {((sum(actual_usability)*100)/(7*len(actual_usability))):.1f}")
    print(f"Usability NPS: {(((promoters - detractors)*100)/len(actual_usability)):.0f}")
    print(f"Average self assessment %: {((sum(actual_self_rating)*100)/(5*len(actual_self_rating))):.1f}")
    
    calc_percentiles(read_time, "Total Read Time")
    for page in inst_timing.keys():
        calc_percentiles(inst_timing[page], f"Page #{page} Read Time")
    calc_percentiles(pre_interact, "Pre-Interaction Time")
    calc_percentiles(interact_time, "Interaction Time")
    calc_percentiles(pre_first_command_time, "Time After Start Before First Command")
    calc_percentiles(post_last_command_time, "Time After Last Command Before End")
    calc_percentiles(rating_time, "Time After Interaction End")
    calc_percentiles(command_timing['total'], "Avg Command Time")
    for order in command_times_by_order.keys():
        calc_percentiles(command_times_by_order[order], f"Command #{order} Time")

    
    usability.sort()
    keys = range(len(usability))
    u_dict = dict(zip(keys, usability))
    plot_hist(u_dict, xlabel="", ylabel="Usability Score", ymax=7)
    self_rating.sort()
    keys = range(len(self_rating))
    s_dict = dict(zip(keys, self_rating))
    plot_hist(s_dict, xlabel="", ylabel="Self Rated Performance Score", ymax=5)
    read_time = [360 if x>360 else x for x in read_time]
    read_time.sort()
    keys = range(len(read_time))
    r_dict = dict(zip(keys, read_time))
    plot_hist(r_dict, target_val=180, xlabel="", ylabel="Instructions Read Time (sec)")
    pre_interact = [100 if x>100 else x for x in pre_interact]
    pre_interact.sort()
    keys = range(len(pre_interact))
    p_dict = dict(zip(keys, pre_interact))
    plot_hist(p_dict, target_val=30, xlabel="", ylabel="Time between instructions and interaction start (sec)")
    interact_time.sort()
    keys = range(len(interact_time))
    i_dict = dict(zip(keys, interact_time))
    plot_hist(i_dict, target_val=300, xlabel="", ylabel="Interaction time (sec)")
    command_num.sort()
    keys = range(len(command_num))
    c_dict = dict(zip(keys, command_num))
    plot_hist(c_dict, target_val=5, xlabel="", ylabel="Number of commands per HIT")

    inst_timing[5] = inst_timing[5][-100:]  # This is a hack until I can find the bug

    for page in inst_timing.keys():
        inst_timing[page] = [0 if x<0 else x for x in inst_timing[page]]
        inst_timing[page] = [90 if x>90 else x for x in inst_timing[page]]
        inst_timing[page].sort()
        keys = range(len(inst_timing[page]))
        page_dict = dict(zip(keys, inst_timing[page]))
        plot_hist(page_dict, xlabel="", ylabel=f"Page {page} read time (sec)")
    for status in command_timing.keys():
        command_timing[status] = [0 if x<0 else x for x in command_timing[status]]
        command_timing[status] = [50 if x>50 else x for x in command_timing[status]]
        command_timing[status].sort()
        keys = range(len(command_timing[status]))
        command_dict = dict(zip(keys, command_timing[status]))
        plot_hist(command_dict, xlabel="", ylabel=f"Command {status} time (sec)")
    '''

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
def compare_worker_ids(total, read_instructions, timer_off, submit_command, logs):
    from collections import Counter
    all_lists = [total, read_instructions, timer_off, submit_command, logs]
    list_names = ["'Full List'", "'Read Instructions'", "'Turned Timer Off'", "'Submitted a command'", "'Recorded S3 Logs'"]
    print(f"Total number of workers: {len(total)}")
    print(f"Worker HIT count: {Counter(total)}")
    print(f"Number of workers who read instructions: {len(read_instructions)}")
    print(f"Number of workers who turned the timer off: {len(timer_off)}")
    print(f"Number of workers who submitted a command (recorded by Mephisto): {len(submit_command)}")
    print(f"Number of workers who submitted logs to S3: {len(logs)}")
    print(f"Number of Workers who submitted a job but did not read instructions: {len([x for x in total if x not in read_instructions])}")
    print(f"Number of Workers who submitted a job but did not turn the timer off: {len([x for x in total if x not in timer_off])}")
    print(f"Number of Workers who submitted a job but did not submit a command: {len([x for x in total if x not in submit_command])}")
    print(f"Number of Workers who submitted a job but did not log to S3: {len([x for x in total if x not in logs])}")
    print(f"Number of Workers who read instructions but did not turn the timer off: {len([x for x in read_instructions if x not in timer_off])}")
    print(f"Number of Workers who read instructions but did not dubmit a command: {len([x for x in read_instructions if x not in submit_command])}")
    print(f"Number of Workers who turned the timer off but did not submit a command: {len([x for x in timer_off if x not in submit_command])}")
    print(f"Number of Workers who turned the timer off but did not log to S3: {len([x for x in timer_off if x not in logs])}")
    print(f"Number of Workers who submitted a command but did not log to S3: {len([x for x in submit_command if x not in logs])}")
    print(f"Number of Workers who submitted a log to S3 but did not submit a command: {len([x for x in logs if x not in submit_command])}")

    for i,l in enumerate(all_lists):
        d1 = Counter(l)
        #print(f"Worker dict {list_names[i]}: {d1}")
        for j,k in enumerate(all_lists):
            d2 = Counter(k)
            for key in d1.keys():
                try:
                    if d1[key] > d2[key]:
                        print(f"{key} appears in {list_names[i]} {d1[key] - d2[key]} times more than in {list_names[j]}")
                except:
                    print(f"{key} appears in {list_names[i]} but not {list_names[j]}")

    

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
    #plt.xticks(np.arange(len(list(dictionary.keys()))), list(dictionary.keys()), rotation='vertical')
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

    get_stats(list(all_turk_interactions["command"]))
    # return all commands as a list
    return list(set(all_turk_interactions["command"]))

#%%
read_s3_bucket("/private/home/ethancarlson/.hitl/20211112115924/turk_logs", "/private/home/ethancarlson/.hitl/parsed/20211112115924")
print("\nNSP Outputs: ")
read_turk_logs("/private/home/ethancarlson/.hitl/parsed/20211112115924", "nsp_outputs")
print("\nError Details: ")
read_turk_logs("/private/home/ethancarlson/.hitl/parsed/20211112115924", "error_details")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # User needs to provide file I/O paths
    parser.add_argument(
        "--turk_logs_directory",
        default="/private/home/ethancarlson/.hitl/20211025173851/turk_logs",
        help="where to read s3 logs from eg. ~/turk_interactions_with_agent",
    )
    parser.add_argument(
        "--parsed_output_directory",
        default="/private/home/ethancarlson/.hitl/parsed/20211025173851",
        help="where to write the collated NSP outputs eg. ~/parsed_turk_logs",
    )
    parser.add_argument(
        "--filename",
        default="nsp_outputs",
        help="name of the CSV file we want to read, eg. nsp_outputs",
    )
    args = parser.parse_args()
    read_s3_bucket(args.turk_logs_directory, args.parsed_output_directory)
    read_turk_logs(args.parsed_output_directory, args.filename)
