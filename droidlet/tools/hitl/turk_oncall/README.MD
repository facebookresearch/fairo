# README

## TAO Smart Bug Report 
- Updated June 17, 2022

### How TAO Smart Bug Report Works
The TAO smark bug report module are mainly composed of two component:
1. TAO Log Listener class - which listens to the status file for a TAO job batch, and initiates a bug report job once logs are ready. 
2. TAO Bug Report Job class - which reads in the logs for a TAO job batch, extracts traceback, and outputs the content of the traceback and the frequency of the traceback to s3.

The relationship between the above classes and the Oncall Job class are shown as the figure below:
<img width="1819" alt="TAO Smart Bug Report Module Structure" src="https://user-images.githubusercontent.com/51009396/173457822-47ce8b2e-6d53-4b44-8c02-dd0cd0c019b3.png">

The logic for extracting traceback are based on the pattern, the traceback are located in between either a line start with date (YYYY-MM-DD) or a line starts with logging keywords(DEBUG, INFO, WARNING, ERROR, and CRITICAL). Thus anything in between lines with such pattern are read in to a string, and if this block is a traceback (i.e. line starts with "Traceback"), it would be included in the bug report output. 

After extracting traceback blocks, the frequcy of each unique traceback is also recorded. A unique trackback is determined by checking the content of the traceback via string comparision. 

*Updated June 17*: 
The chat contents associated with the traceback is also recorded in the bug report, each traceback can have mutiple chats, thus chat are stored in a list and output into the final csv file. 

For processing logs from past runs, you can use the following script:
```
python oncall_bug_report_offline.py --start_batch_id <START_BATCH_ID> --end_batch_id <END_BATCH_ID>
```
This will crawl s3 storage for any batch id in between range [START_BATCH_ID, END_BATCH_ID] and generates bug report for the batches with ids fell into the range.

### Updated Files
- Added:
    1. `droidlet/tools/hitl/turk_oncall/oncall_bug_report.py`
    2. `droidlet/tools/hitl/turk_oncall/tests/oncall_job_mock.py`
    3. `droidlet/tools/hitl/turk_oncall/oncall_bug_report_offline.py`
- Updated:
    1. `droidlet/tools/hitl/turk_oncall/oncall_jobs.py`

### Run Test
For testing the smart bug report, please follow steps shown as below:

0. (Setup, skip this if you have setup the development environment before) Follow the setup guide under droidlet folder
1. Find a batch id (or multiple batch ids) of a TAO job batch
2. With the batch id(s), run the following command:
    
    ```
    $ python oncall_bug_report.py --tao_job_batch_id [test_batch_id1] [test_batch_id2] [test_batch_id3]...
    ```

The output log file would be located on s3:
```
s3://droidlet-hitl/{test_batch_id}/log_traceback.csv 
```
An example log_traceback.csv is as follow:
<img width="1256" alt="Example log_traceback.csv" src="https://user-images.githubusercontent.com/51009396/174422200-6710551c-98ee-448c-9ce9-120de37140e8.png">
