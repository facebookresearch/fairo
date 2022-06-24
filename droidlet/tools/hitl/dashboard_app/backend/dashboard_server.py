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
socketio = SocketIO(app, cors_allowed_origins="*")


class DASHBOARD_EVENT(Enum):
    """
    server supported event types
    """

    GET_JOBS = "get_job_list"
    GET_TRACEBACK = "get_traceback_by_id"
    GET_RUN_INFO = "get_run_info_by_id"


@socketio.on(DASHBOARD_EVENT.GET_JOBS.value)
def get_jobs():
    print(f"Request received: {DASHBOARD_EVENT.GET_JOBS.value}")
    job_list = get_job_list()
    print(f"Job list reterived from aws, sending job list (length:{len(job_list)}) to client")
    emit(DASHBOARD_EVENT.GET_JOBS.value, job_list)


@socketio.on(DASHBOARD_EVENT.GET_TRACEBACK.value)
def get_traceback(job_id):
    print(f"Request received: {DASHBOARD_EVENT.GET_TRACEBACK.value}")
    log_content = get_traceback_by_id(int(job_id))
    emit(DASHBOARD_EVENT.GET_TRACEBACK.value, log_content)


@socketio.on(DASHBOARD_EVENT.GET_RUN_INFO.value)
def get_info(job_id):
    print(f"Request received: {DASHBOARD_EVENT.GET_RUN_INFO.value}")
    run_info = get_run_info_by_id(int(job_id))
    emit(DASHBOARD_EVENT.GET_RUN_INFO.value, run_info)


if __name__ == "__main__":
    socketio.run(app)
