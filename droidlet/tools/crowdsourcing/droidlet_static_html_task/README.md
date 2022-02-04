## Craftassist Interaction HITs

This folder contains the scripts and files necessary to run the Craftassist interaction HITs standalone or as part of the HiTL pipeline, including the main interaction task and the qualification pilot task to build the interaction job whitelist.

### Selected Folder and File Descriptions

 - `static_test_script.py` is for testing a task on a local node server and usually runs based off the `example.yaml` config file
 - `static_run_qith_qual.py` launches the main interaction task based on `run_with_qual.yaml`
 - `process_s3_logs.py` collects interaction session logs from S3 and postprocesses them
 - `pilot_config.py` contains the allowlist and blocklist names for import elsewhere
 - `pilot_task_run.py` launches the HIT to build the allowlist
 - `issue_bonus.py` will issue performance incentive bonuses for all of the HITs associated with a task_name, unless bonuses have already been issued for those HITs and recorded in S3
 - `examine_results.py` allows for the viewing of task data recorded in the local Mephisto database associated with the provided task_name

 - `conf` contains the .yaml config files for HITs.  This is where the job knows which html files to serve, which data.csv contains the HIT input data, and various other HIT parameters such as payment amount, task name, and timeout length.
 - `server_files` contains the .html and associated ancillary resources (in `extra_refs`) that are served by Mephisto in the HIT page


### Interaction HIT User Story

Users are presented with the /turk.js endpoint of the [dashboard](https://github.com/facebookresearch/fairo/tree/main/droidlet/dashboard/web).

![dashboard image](https://craftassist.s3.us-west-2.amazonaws.com/pubr/mturk_dashboard_new.png)

After reading the instructions and pressing the start button, workers are instructed to interact with the agent for 5 minutes, issuing text commands through a chat box.  On the top right of the screen there is a window where workers can view and navigate a voxel-based world similar to Minecraft that they share with the assistant bot. On the bottom of the screen there is a "stoplight" which serves as visual feedback of the quality of the interaction.  Based on the number of commands issued (more is better), the diversity of commands issued from each other, and the creativity of commands compared to the corpus of previous commands, the stoplight will change from red to yellow to green.  Workers are also given a text prompt below the stoplight how they might improve their score.

Internally, the interaction is scored out of 10, and the stoplight is a direct function of that score.  Workers are paid a bonus based on the score they end up with at the end of the HIT, currently $0.30 per point for a maximum of $3.00.

At the end of the HIT, workers are asked to rate the usability of the HIT and provide any suggestions for improvement, as well as rate the quality of their interaction.  Some research has shown that when crowd workers are asked to reflect on their own performance it encourages them to improve their work over time.
