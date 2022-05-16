## Turk-As-Oncall HITs

This folder contains the files necessary to run the a special version of the interaction jobs that is used to test the functionality of the agent and check for bugs.

### Selected Folder and File Descriptions

 - `static_test_script.py` is for testing a task on a local node server and usually runs based off the `example.yaml` config file
 - `static_run_qith_qual.py` launches the main oncall task based on `run_with_qual.yaml`
 - `examine_results.py` allows for the viewing of task data recorded in the local Mephisto database associated with the provided task_name

 - `hydra_configs/conf` contains the .yaml config files for HITs.  This is where the job knows which html files to serve, which data.csv contains the HIT input data, and various other HIT parameters such as payment amount, task name, and timeout length.
 - `server_files` contains the .html and associated ancillary resources (in `extra_refs`) that are served by Mephisto in the HIT page


### Interaction HIT User Story

Users are presented with the /turk.js endpoint of the [dashboard](https://github.com/facebookresearch/fairo/tree/main/droidlet/dashboard/web).

![dashboard image](https://craftassist.s3.us-west-2.amazonaws.com/pubr/mturk_dashboard_new.png)

After reading the instructions and pressing the start button, workers are instructed to send a specific set ~3 commands.  Next to each command is listed the expected agent behavior associated with that command based on the current agent capabilities.  After each command, the user marks in a survey whether or not the agent actually carried out the expected behavior.