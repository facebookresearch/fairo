import subprocess
import time
import sys

"""
Kicks off a pipeline that schedules Turk jobs for tool 1B,
collects results in batches and collates data.

1. Read in newline separated commands and construct CSV input.
2. Create HITs for each input command using tool 1B template.
3. Continuously check for completed assignments, fetching results in batches.
4. Collate turk output data with the input job specs.
5. Postprocess datasets to obtain well formed action dictionaries.
"""

dev_flag = ""
# Parse flags passed into script run
if len(sys.argv) > 1:
    # flag to toggle dev mode is --dev
    dev_flag = sys.argv[1]

# CSV input
rc = subprocess.call(
    [
        "python3 construct_input_for_turk.py --input_file C/input.txt --tool_num 3 > C/turk_input.csv"
    ],
    shell=True,
)
if rc != 0:
    print("Error preprocessing. Exiting.")
    sys.exit()

# Load input commands and create a separate HIT for each row
rc = subprocess.call(
    [
        "python3 create_jobs.py --xml_file fetch_question_C.xml --tool_num 3 --input_csv C/turk_input.csv --job_spec_csv C/turk_job_specs.csv {}".format(dev_flag)
    ],
    shell=True,
)
if rc != 0:
    print("Error creating HIT jobs. Exiting.")
    sys.exit()
# Wait for results to be ready
print("Turk jobs created for tool C at : %s \n Waiting for results..." % time.ctime())
print("*"*50)

time.sleep(200)
# Check if results are ready
rc = subprocess.call(
    [
        "python3 get_results.py --output_csv C/turk_output.csv {}".format(dev_flag)
    ],
    shell=True,
)
if rc != 0:
    print("Error fetching HIT results. Exiting.")
    sys.exit()

# Collate datasets
print("*"*50)
print("*** Collating turk outputs and input job specs ***")
rc = subprocess.call(["python3 collate_answers.py --turk_output_csv C/turk_output.csv --job_spec_csv C/turk_job_specs.csv --collate_output_csv C/processed_outputs.csv"], shell=True)
if rc != 0:
    print("Error collating answers. Exiting.")
    sys.exit()


# Postprocess
print("*"*50)
print("*** Postprocessing results ***")
rc = subprocess.call(["python3 parse_tool_C_outputs.py"], shell=True)
if rc != 0:
    print("Error collating answers. Exiting.")
    sys.exit()

# Create inputs for other tools
print("*"*50)
print("*** Postprocessing results and generating input for tool D***")
rc = subprocess.call(["python3 generate_input_for_tool_D.py"], shell=True)
if rc != 0:
    print("Error generating input for other tools. Exiting.")
    sys.exit()
print("*"*50)