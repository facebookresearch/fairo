# Dashboard App for HITL
Updated Jun 24 by Chuxi.

This is a Dashboard app prototype for the HITL system.

## Included 
- Backend: a backend based on flask and socket. Run on localhost:5000 by default.
- Frontend: a react based frontend - currently only have button for user interaction. Defult run on localhost:3000.
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
        meta data like name of the run, batch id, start time/end time, stastics for each HIT jobs in this run, etc. 
    - input: a batch id.
    - output: if the run info can be found, return the run info in a json format, otherwise, return an error message sugesting not found.

## Demo
