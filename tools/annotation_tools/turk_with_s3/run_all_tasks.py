import subprocess
import time
import sys
import os

"""
Kicks off a pipeline that schedules Turk jobs for tool 1A,
collects results in batches and collates data.

Tool A run finishes
-> run tool B
-> run tool C
-> run tool D
"""
dev_flag = ""
# Parse flags passed into script run
if len(sys.argv) > 1:
    # flag to toggle dev mode is --dev
    dev_flag = sys.argv[1]

# Default to directory of script being run for writing inputs and outputs
default_write_dir = os.path.dirname(os.path.abspath(__file__))

# Creating directories for tool outputs
folder_name_A = '{}/A/'.format(default_write_dir)
folder_name_B = '{}/B/'.format(default_write_dir)
folder_name_C = '{}/C/'.format(default_write_dir)
folder_name_D = '{}/D/'.format(default_write_dir)

# If the tool specific write directories do not exist, create them
if not os.path.isdir(folder_name_A):
    os.mkdir(folder_name_A)

if not os.path.isdir(folder_name_B):
    os.mkdir(folder_name_B)

if not os.path.isdir(folder_name_C):
    os.mkdir(folder_name_C)

if not os.path.isdir(folder_name_D):
    os.mkdir(folder_name_D)

# CSV input
rc = subprocess.call(
    [
        "python3 run_tool_1A.py {}".format(dev_flag)
    ],
    shell=True,
)
if rc != 0:
    print("Error preprocessing. Exiting.")
    sys.exit()

# If the tool B input file is not empty, kick off a job
if os.path.exists("B/input.txt") and os.path.getsize("B/input.txt") > 0:
    # Load input commands and create a separate HIT for each row
    rc = subprocess.call(["python3 run_tool_1B.py {}".format(dev_flag)], shell=True)
    if rc != 0:
        print("Error creating HIT jobs. Exiting.")
        sys.exit()
    # Wait for results to be ready
    print("Turk jobs created at : %s \n Waiting for results..." % time.ctime())

# If the tool C input file is not empty, kick off a job
if os.path.exists("C/input.txt") and os.path.getsize("C/input.txt") > 0:
    # Check if results are ready
    rc = subprocess.call(["python3 run_tool_1C.py {}".format(dev_flag)], shell=True)
    if rc != 0:
        print("Error fetching HIT results. Exiting.")
        sys.exit()

# If the tool B input file is not empty, kick off a job
if os.path.exists("D/input.txt") and os.path.getsize("D/input.txt") > 0:
    # Collate datasets
    print("*** Collating turk outputs and input job specs ***")
    rc = subprocess.call(["python3 run_tool_1D.py {}".format(dev_flag)], shell=True)
    if rc != 0:
        print("Error collating answers. Exiting.")
        sys.exit()

# Run final postprocessing on the action dictionaries to construct logical forms in sync with grammar
rc = subprocess.call(["python3 construct_final_action_dict.py"], shell=True)
if rc != 0:
    print("Error constructing final dictionary. Exiting.")
    sys.exit()