import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--default_write_dir", type=str, required=True)
parser.add_argument("--timeout", type=float, default=60)
parser.add_argument("--dev", default=False, action="store_true")

args = parser.parse_args()
dev_flag = "--dev" if args.dev else ""
default_write_dir = args.default_write_dir
timeout = 300#args.timeout

# # Default to directory of script being run for writing inputs and outputs
# default_write_dir = os.path.dirname(os.path.abspath(__file__))

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

print("Running A ... ")
# CSV input
rc = subprocess.call(
    [
        f"python3 run_tool_1A.py {dev_flag} --default_write_dir {default_write_dir} --timeout {timeout}"
    ],
    shell=True,
)
if rc != 0:
    print("Error preprocessing. Exiting.")
    sys.exit()

print("Running B ... ")
# If the tool B input file is not empty, kick off a job
if os.path.exists(f"{default_write_dir}/B/input.txt") and os.path.getsize(f"{default_write_dir}/B/input.txt") > 0:
    # Load input commands and create a separate HIT for each row
    rc = subprocess.call([f"python3 run_tool_1B.py  --default_write_dir {default_write_dir} {dev_flag} --timeout {timeout}"], shell=True)
    if rc != 0:
        print("Error creating HIT jobs. Exiting.")
        sys.exit()

print("Running C ... ")
# If the tool C input file is not empty, kick off a job
if os.path.exists(f"{default_write_dir}/C/input.txt") and os.path.getsize(f"{default_write_dir}/C/input.txt") > 0:
    # Check if results are ready
    rc = subprocess.call([f"python3 run_tool_1C.py --default_write_dir {default_write_dir} {dev_flag} --timeout {timeout}"], shell=True)
    if rc != 0:
        print("Error fetching HIT results. Exiting.")
        sys.exit()

print("Running D ... ")
# If the tool B input file is not empty, kick off a job
if os.path.exists(f"{default_write_dir}/D/input.txt") and os.path.getsize(f"{default_write_dir}/D/input.txt") > 0:
    # Collate datasets
    print("*"*40)
    print("*** Collating turk outputs and input job specs ***")
    rc = subprocess.call([f"python3 run_tool_1D.py --default_write_dir {default_write_dir} {dev_flag} --timeout {timeout}"], shell=True)
    if rc != 0:
        print("Error collating answers. Exiting.")
        sys.exit()

# Run final postprocessing on the action dictionaries to construct logical forms in sync with grammar
rc = subprocess.call([f"python3 construct_final_action_dict.py --write_dir_path {default_write_dir}"], shell=True)
if rc != 0:
    print("Error constructing final dictionary. Exiting.")
    sys.exit()