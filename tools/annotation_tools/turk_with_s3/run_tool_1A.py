import subprocess
import time
import sys

"""
Kicks off a pipeline that schedules Turk jobs for tool 1A,
collects results in batches and collates data.

1. Read in newline separated commands and construct CSV input.
2. Create HITs for each input command using tool 1A template.
3. Continuously check for completed assignments, fetching results in batches.
4. Collate turk output data with the input job specs.
5. Postprocess datasets to obtain well formed action dictionaries.
"""

# CSV input
rc = subprocess.call(
    [
        "python3 ../text_to_tree_tool/construct_input_for_turk.py --input_file input.txt > A/turk_input.csv"
    ],
    shell=True,
)
if rc != 0:
    print("Error preprocessing. Exiting.")
    sys.exit()

# Load input commands and create a separate HIT for each row
rc = subprocess.call(["python3 create_jobs.py --tool_num 1 --xml_file step_1.xml --input_csv A/turk_input.csv --job_spec_csv A/turk_job_specs.csv"], shell=True)
if rc != 0:
    print("Error creating HIT jobs. Exiting.")
    sys.exit()
# Wait for results to be ready
print("Turk jobs created at : %s \n Waiting for results..." % time.ctime())

time.sleep(100)
# Check if results are ready
rc = subprocess.call(["python3 get_results.py --output_csv A/turk_output.csv"], shell=True)
if rc != 0:
    print("Error fetching HIT results. Exiting.")
    sys.exit()

# Collate datasets
print("*** Collating turk outputs and input job specs ***")
rc = subprocess.call(["python3 collate_answers.py --turk_output_csv A/turk_output.csv --job_spec_csv A/turk_job_specs.csv --collate_output_csv A/processed_outputs.csv"], shell=True)
if rc != 0:
    print("Error collating answers. Exiting.")
    sys.exit()

# Postprocess
print("*** Postprocessing results ***")
rc = subprocess.call(["python3 parse_tool_A_outputs.py"], shell=True)
if rc != 0:
    print("Error collating answers. Exiting.")
    sys.exit()

# Create inputs for other tools
print("*** Postprocessing results ***")
rc = subprocess.call(["python3 generate_input_for_tool_B_and_C.py"], shell=True)
if rc != 0:
    print("Error generating input for other tools. Exiting.")
    sys.exit()