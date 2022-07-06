# Dashboard App for HITL
Updated July 5 by Chuxi.

This is a Dashboard app prototype for the HITL system.

## Update Notes
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

## Demo
![backend_api_demo](https://user-images.githubusercontent.com/51009396/175696481-532cec55-5b2e-4bae-bceb-9e7d3f2aa7b7.gif)
