import subprocess
import time
import sys

"""
Kicks off a pipeline that schedules Turk jobs for tool 1A,
collects results in batches and collates data.

1. Read in newline separated commands and construct CSV input.
2. Create HITs for each input command using tool combined template.
3. Continuously check for completed assignments, fetching results in batches.
4. Collate turk output data with the input job specs.
5. Write Turk output.
6. Create input for turk job A.

NOTE: consider converting these scripts to python functions in the future.
That would be more sustainable eg. to run unit tests.
"""

dev_flag = ""
# Parse flags passed into script run
if len(sys.argv) > 1:
    # flag to toggle dev mode is --dev
    dev_flag = sys.argv[1]

# CSV input
rc = subprocess.call(
    [
        "python construct_input_for_turk.py --input_file input_combined.txt --tool_num 6 > turk_input_combined.csv"
    ],
    shell=True,
)
if rc != 0:
    print("Error preprocessing. Exiting.")
    sys.exit()

# Load input commands and create a separate HIT for each row
rc = subprocess.call(
    [
        "python create_jobs.py --xml_file fetch_combined_tool.xml --input_csv turk_input_combined.csv --job_spec_csv turk_job_combined.csv {}".format(dev_flag)
    ],
    shell=True,
)
if rc != 0:
    print("Error creating HIT jobs. Exiting.")
    sys.exit()
# Wait for results to be ready
print("Turk jobs created for tool combined at : %s \n Waiting for results..." % time.ctime())
print("*"*50)

time.sleep(100)
# Check if results are ready
rc = subprocess.call(
    [
        "python get_results.py --output_csv turk_combined_output.csv {}".format(dev_flag)
    ],
    shell=True,
)
if rc != 0:
    print("Error fetching HIT results. Exiting.")
    sys.exit()

# Collate datasets
print("*"*50)
print("*** Collating turk outputs and input job specs ***")
rc = subprocess.call(["python collate_answers.py --turk_output_csv turk_combined_output.csv --job_spec_csv turk_job_combined.csv --collate_output_csv processed_outputs_combined.csv"], shell=True)
if rc != 0:
    print("Error collating answers. Exiting.")
    sys.exit()

# Postprocess
print("*"*50)
print("*** Postprocessing results ***")
rc = subprocess.call(["python parse_tool_combined.py"], shell=True)
if rc != 0:
    print("Error postprocessing answers and generating input for A. Exiting.")
    sys.exit()
