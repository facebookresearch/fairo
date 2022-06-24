"""
Copyright (c) Facebook, Inc. and its affiliates.

This file include a flask server that is the backend of the HITL dashboard app.
"""


from enum import Enum
import logging
from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
    get_job_list,
    get_run_info_by_id,
    get_traceback_by_id,
)
from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*") # allow cors


class DASHBOARD_EVENT(Enum):
    """
    server supported event types, i.e. API types
    """

    GET_JOBS = "get_job_list"
    GET_TRACEBACK = "get_traceback_by_id"
    GET_RUN_INFO = "get_run_info_by_id"


@socketio.on(DASHBOARD_EVENT.GET_JOBS.value)
def get_jobs():
    """
    get a list of jobs stored on AWS that has been run in the past. 
    - input: no parameter input.
    - output: a list of batch ids of the jobs.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_JOBS.value}")
    job_list = get_job_list()
    print(f"Job list reterived from aws, sending job list (length:{len(job_list)}) to client")
    emit(DASHBOARD_EVENT.GET_JOBS.value, job_list)


@socketio.on(DASHBOARD_EVENT.GET_TRACEBACK.value)
def get_traceback(batch_id):
    """
    get traceback record by id.
    - input: a batch id.
    - output: if the traceback record can be found, return the traceback in csv format, otherwise, output an error message suggesting not found.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_TRACEBACK.value}")
    log_content = get_traceback_by_id(int(batch_id))
    emit(DASHBOARD_EVENT.GET_TRACEBACK.value, log_content)


@socketio.on(DASHBOARD_EVENT.GET_RUN_INFO.value)
def get_info(batch_id):
    """
    get run info by id, run info could be:
        meta data like name of the run, batch id, start time/end time, stastics for each HIT jobs in this run, etc. 
    - input: a batch id.
    - output: if the run info can be found, return the run info in a json format, otherwise, return an error message sugesting not found.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_RUN_INFO.value}")
    run_info = get_run_info_by_id(int(batch_id))
    emit(DASHBOARD_EVENT.GET_RUN_INFO.value, run_info)


if __name__ == "__main__":
    socketio.run(app)
