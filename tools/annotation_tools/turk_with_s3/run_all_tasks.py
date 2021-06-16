import subprocess
import time
import sys

"""
Kicks off a pipeline that schedules Turk jobs for tool 1A,
collects results in batches and collates data.

Tool A run finishes
-> run tool B
-> run tool C
-> run tool D
"""

# CSV input
rc = subprocess.call(
    [
        "python3 run_tool_1A.py"
    ],
    shell=True,
)
if rc != 0:
    print("Error preprocessing. Exiting.")
    sys.exit()

# Load input commands and create a separate HIT for each row
rc = subprocess.call(["python3 run_tool_1B.py"], shell=True)
if rc != 0:
    print("Error creating HIT jobs. Exiting.")
    sys.exit()
# Wait for results to be ready
print("Turk jobs created at : %s \n Waiting for results..." % time.ctime())

time.sleep(100)
# Check if results are ready
rc = subprocess.call(["python3 run_tool_1C.py"], shell=True)
if rc != 0:
    print("Error fetching HIT results. Exiting.")
    sys.exit()

# Collate datasets
print("*** Collating turk outputs and input job specs ***")
rc = subprocess.call(["python3 run_tool_1D.py"], shell=True)
if rc != 0:
    print("Error collating answers. Exiting.")
    sys.exit()