# Dashboard App for HITL
Updated Aug 10 by Chuxi.

This is a Dashboard app prototype for the HITL system.

## Update Note
- Aug 10:
    - Updated dashboard_aws_helper_test.py to use mock s3. 
- July 14:
    - Added model visualization component, user can see text info about a model:
    - Demo:
        - View a model for a run with retrain jobs:
            - ![demo_view_model_with_retrain](https://user-images.githubusercontent.com/51009396/179124247-e1521123-aeaa-4ea9-a47a-d05323a37d3a.gif)
        - Model showing NA for a run without retrain jobs:
            - ![demo_view_model_no_retrain](https://user-images.githubusercontent.com/51009396/179124182-9840f1ff-ccbe-4709-9452-025fb5be5c8a.gif) 

- July 13:
    - Added dataset visualization component including:
        - View newly added data points indices on the run detail page
        - View a specific version of dataset on dataset detail page
    - Demo:
        - For job with retrain job, the user will be able to see the indices of newly added data
            - ![demo_dataset_has_retrain](https://user-images.githubusercontent.com/51009396/178807612-89f8d3cb-3555-45be-9bc1-20349129eb11.gif)- 
        - For job without retrain job, the indices field will be NA
            - ![demo_dataset_no_retrain](https://user-images.githubusercontent.com/51009396/178807684-4bc5544e-c8a8-4322-a46c-ed5cdc3345ac.gif)
        - Viewing different version of the dataset (use dropdown menu to select, able to serach in dropdown menu)
            -  ![demo_change_dataset_ver](https://user-images.githubusercontent.com/51009396/178807722-62baf30a-9c0f-4935-8061-098a5f90ab4f.gif)
        - Flow for using the navigation tab to view a dataset
            - ![demo_navigate_from_tab](https://user-images.githubusercontent.com/51009396/178807807-68425eaa-418a-4128-bc9c-076d8aef3fd5.gif)
- July 7:
    - Added view session Log feature.
    - Demo:
        - ![demo_view_sesion_log](https://user-images.githubusercontent.com/51009396/178044751-8d099829-cc2f-4353-8bfd-e81e9a23b63e.gif)
- July 6: ![demo_change_dataset_ver](https://user-images.githubusercontent.com/51009396/178807740-b90ada63-573d-47ba-845e-8ab84033d184.gif)
    - Added detail page to show detail infomation of a pipeline run (specified by the batch_id)
    - Added a get all sessions related to a batch id backend API. 
    - Demo:
        - For a batch_id that is associate with detail info, the user will see the following detail page:
            - ![detail_valid_batch_id](https://user-images.githubusercontent.com/51009396/177655919-693870cc-595c-416e-ac25-651e3cccbeca.gif)
        - For a batch_id that has no detail info, will redirect to not found page.
            - ![detail_not_vaild_batch_id](https://user-images.githubusercontent.com/51009396/177655956-c398f669-90af-4a13-bdff-736057ca68cb.gif)
- July 5: added search & filter for the NLU run list. 
    - User can search/filter the run list, and also clear search/filter to view the completed run list. 
    - Search demo:
        - ![search_demo](https://user-images.githubusercontent.com/51009396/177654353-cf34635d-9571-4666-83c3-b9b33afde9a8.gif)
    - Filter demo:
        - ![filter_demo](https://user-images.githubusercontent.com/51009396/177654387-a660f9ae-c519-4e21-8c4c-07af9aa6cf76.gif)
        
## Included 
- Backend: a backend based on flask and socket. Run on localhost:5000 by default.
- Frontend: a react based frontend - including navigation and NLU pipeline overview page. Default run on localhost:3000.
Please note this dashboard app is currently only in development build and is not optimized.

## Dependency
- Dependency required for backend server (in addition to the droidelet setting, please make sure you have the updated version as below):
Flask==2.1.2
Flask-SocketIO==5.2.0
- You need to have the following python package to be able to run the tests:
moto==3.1.17
- Dependency for frontend is specified in the `fairo/droidlet/tools/hitl/dashboard_app/dashboard_frontend/package.json` file.

## How to run
- Set up environment variables
```
export AWS_ACCESS_KEY_ID={your aws key id}
export AWS_DEFAULT_REGION={your aws region}
export AWS_SECRET_ACCESS_KEY={your aws secret acess key}
```

- Backend:
```
cd backend
python dashboard_server.py
```

- Frontend:
```
cd dashboard_frontend
npm install (first time setup)
npm start
```

## Supported APIs
APIs are based on socket event, the following APIs are supported currently: 
- get_job_list:
    - get a list of jobs stored on AWS that has been run in the past. 
    - input: no parameter input.
    - output: a list of batch ids of the jobs.
- get_traceback_by_id:
    - get traceback record by id.
    - input: a batch id.
    - output: if the traceback record can be found, return the traceback in csv format, otherwise, output an error message suggesting not found.
- get_run_info_by_id:
    - get run info by id, run info could be:
        meta data like name of the run, batch id, start time/end time, statistics for each HIT jobs in this run, etc. 
    - input: a batch id.
    - output: if the run info can be found, return the run info in a json format, otherwise, return an error message sugesting not found.
- get_interaction_sessions_by_id
    - get interaction job sessions list
    - input: a batch id.
    - output: if the sessions can be found, return a list of session name, otherwise, return an error code sugesting not found.
- get_interaction_session_log
    - get interaction job session log specified by the id info
    - input: infomation about id in a json format:
        ```json
        {   
            "batch_id": <batch id>,
            "session_id": <session id>
        }
        ```
    - output: if the session log can be found, return the content of session log, otherwise return an error code
- get_dataset_list_by_pipeleine
    - get pipeline specific dataset list
    - input: the pipeline name.
    - output: the list of dataset used in the specified pipeline
- get_dataset_by_name
    - get specific version of dataset
    - input: the name of the dataset.
    - output: if the dataset can be found, return the dataset content, otherwise return an error code
- get_dataset_idx_by_id
    - get run specific dataset indices: as for each of the run, more data point can be added to the dataset, the indices specified the start index and the end index of the data points added to the dataset in a given run
    - input: the batch id of the run.
    - output: [start_index, end_index] of the data added to the dataset with the specified run or error code if cannot find the meta.txt
- get all keys for a model related to a run (specified by the batch_id)
    - input: the batch id of the run.
    - output: the keys for the model if the model exists, otherwise error code
 - get a value for a model related to a run (specified by the batch_id) and the input key
    - input:
        - the batch id of the run.
        - the key for the model, could be any key from the model, or "COMPLETE", indicating getting the complete model dict
    - output: the key and the value specific to the key for the model if the model exists and key is valid, otherwise error code
    
## Demo
![backend_api_demo](https://user-images.githubusercontent.com/51009396/175696481-532cec55-5b2e-4bae-bceb-9e7d3f2aa7b7.gif)
