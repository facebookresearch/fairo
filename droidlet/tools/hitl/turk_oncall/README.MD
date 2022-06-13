# README

## TAO Smart Bug Report 
- Updated June 10, 2022
### How TAO Smart Bug Report Works
The TAO smark bug report module are mainly composed of two component:
1. TAO Log Listener class - which listens to the status file for a TAO job batch, and initiates a bug report job once logs are ready. 
2. TAO Bug Report Job class - which reads in the logs for a TAO job batch, extracts traceback, and outputs the content of the traceback and the frequency of the traceback to s3.

The relationship between the above classes and the Oncall Job class are shown as the figure below:

The logic for extracting traceback are based on the pattern, the traceback are located in between either a line start with date (YYYY-MM-DD) or a line starts with logging keywords(DEBUG, INFO, WARNING, ERROR, and CRITICAL). Thus anything in between lines with such pattern are read in to a string, and if this block is a traceback (i.e. line starts with "Traceback"), it would be included in the bug report output. 

After extracting traceback blocks, the frequcy of each unique traceback is also recorded. A unique trackback is determined by checking the content of the traceback via string comparision. 

### Updated Files
- Added:
    1. `droidlet/tools/hitl/turk_oncall/oncall_bug_report.py`
    2. `droidlet/tools/hitl/turk_oncall/tests/oncall_job_mock.py`
- Updated:
    1. `droidlet/tools/hitl/turk_oncall/oncall_jobs.py`

### Run Test
For testing the smart bug report, please follow steps shown as below:

0. (Setup, skip this if you have setup the development environment before) Follow the setup guide under droidlet folder
1. Find a batch id of a TAO job batch
2. With the batch id, run the following command:
    
    ```
    $ python oncall_bug_report.py --tao_job_batch_id=[test_batch_id]
    ```

The output log file would be located on s3:
```
s3://droidlet-hitl/{test_batch_id}/log_traceback.csv 
```
An example log_traceback.csv is as follow:
  <img width="1173" alt="Example log_traceback.csv" src="https://user-images.githubusercontent.com/51009396/173154746-f8dc5247-77b1-4b70-bb35-4c89804beec4.png">
