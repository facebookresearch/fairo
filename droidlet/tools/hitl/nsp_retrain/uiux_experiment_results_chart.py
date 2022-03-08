#%%
"""
Generate plot of UI/UX experiment results
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

#%%
"""
Data:
X - Experiment number
Y - Average Ratio of NSP Errors per $ over baseline
Method:
 - For each data point, calculate NSP Errors per $
 - Calculate average of all points in each experiment
 - Calculate 2.5% and 97.5% of each experiment for error bars
"""

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
def read_turk_logs(turk_output_directory, filename):

    cnt_list = []
    for csv_path in glob.glob(
        "{turk_logs_dir}/**/{csv_filename}".format(
            turk_logs_dir=turk_output_directory, csv_filename=filename + ".csv"
        )
    ):

        with open(csv_path) as fd:
            # collect the NSP outputs CSV
            csv_file = pd.read_csv(csv_path, delimiter="|")
            csv_file = csv_file.drop_duplicates(subset=["command"])

            cnt_list.append(len(csv_file.index))

    return cnt_list


#%%
def compute_stats(epd, exp):
    avg = np.average(epd)
    stdev = np.std(epd)
    sterr = stdev / ((len(epd)) ** 0.5)

    print(f"{exp} Average: {avg}")
    print(f"{exp} Standard Error: {sterr}")

    return (avg, sterr)


#%%
def plot_hist(xs, ys, errs, ylabel, target_val=None, xlabel="UI/UX Experiments", ymax=None):
    plt.bar(xs, ys, yerr=errs, color="g")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax:
        plt.ylim(0, ymax)
    plt.show()


#%%
# errs / (cost * #logs)
stats = []
exp1_epd = [(7 / (3 * 16)), (65 / (3 * 98))]
stats.append(compute_stats(exp1_epd, "Exp 1"))
exp2_epd = [(172 / 297), (123 / 300)]
stats.append(compute_stats(exp2_epd, "Exp 2"))
exp3_epd = [(171 / 294), (176 / 279)]
stats.append(compute_stats(exp3_epd, "Exp 3"))
exp4_epd = [(133 / 190.14), (140 / 184.3), (113 / 197.75)]
stats.append(compute_stats(exp4_epd, "Exp 4"))

xs = ["Exp1", "Exp2", "Exp3", "Exp4"]
ys = [y[0] for y in stats]
errs = [y[1] for y in stats]

plot_hist(xs, ys, errs, "Data Generation Efficiency (% / Baseline)")

#%%


#%%
"""
Scrape data for Exp 1 from logs
"""
cost_per = 3.00
logs_dirs = [
    "/private/home/ethancarlson/.hitl/parsed/20211025173851",
    "/private/home/ethancarlson/.hitl/parsed/20211027100532",
]
errors_per_dollar = []
for dir in logs_dirs:
    errors_per_dollar += [x / cost_per for x in read_turk_logs(dir, "error_details")]

#%%
"""
Scrape data for Exp 2
"""
run_ids = []
cost_per = 3.00
errors_per_dollar = []
for run_id in run_ids:
    completed_units = retrieve_units(run_id)
    db = LocalMephistoDB()
    data_browser = DataBrowser(db=db)
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        worker = Worker(db, data["worker_id"]).worker_name
        outputs = data["data"]["outputs"]
        clean_click_string = outputs["clickedElements"].replace("'", "")
        clicks = json.loads(clean_click_string)
        parsing_error_count = 0
        if clicks:
            for click in clicks:
                # Count parsing errors
                if "parsing_error" in click["id"]:
                    if click["id"]["parsing_error"]:
                        parsing_error_count += 1
        errors_per_dollar.append(float(parsing_error_count) / cost_per)

print(len(errors_per_dollar))
print(sum(errors_per_dollar) / len(errors_per_dollar))
