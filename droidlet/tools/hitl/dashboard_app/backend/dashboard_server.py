"""
Copyright (c) Facebook, Inc. and its affiliates.

This file include a flask server that is the backend of the HITL dashboard app.
"""


from enum import Enum
import json
from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
    get_dataset_by_name,
    get_dataset_indices_by_id,
    get_dataset_version_list_by_pipeline,
    get_interaction_session_log_by_id,
    get_interaction_sessions_by_id,
    get_job_list,
    get_model_by_id,
    get_run_info_by_id,
    get_traceback_by_id,
)
from droidlet.tools.hitl.dashboard_app.backend.dashboard_model_utils import (
    get_keys,
    get_value_by_key,
)
from flask import Flask, abort
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")  # allow cors


class DASHBOARD_EVENT(Enum):
    """
    server supported event types, i.e. API types
    """

    GET_RUNS = "get_job_list"
    GET_TRACEBACK = "get_traceback_by_id"
    GET_RUN_INFO = "get_run_info_by_id"
    GET_INTERACTION_SESSIONS = "get_interaction_sessions_by_id"
    GET_INTERACTION_SESSION_LOG = "get_interaction_session_log"

    GET_DATASET_LIST = "get_dataset_list_by_pipeleine"
    GET_DATASET_INDECIES = "get_dataset_idx_by_id"
    GET_DATASET = "get_dataset_by_name"
    GET_MODEL_KEYS = "get_model_keys_by_id"
    GET_MODEL_VALUE = "get_model_value_by_id_n_key"


# constants for model related apis
KEY_COMPLETE = "complete_model"


@socketio.on(DASHBOARD_EVENT.GET_RUNS.value)
def get_jobs():
    """
    get a list of jobs stored on AWS that has been run in the past.
    - input: no parameter input.
    - output: a list of batch ids of the jobs.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_RUNS.value}")
    job_list = get_job_list()
    print(f"Job list reterived from aws, sending job list (length:{len(job_list)}) to client")
    emit(DASHBOARD_EVENT.GET_RUNS.value, job_list)


@socketio.on(DASHBOARD_EVENT.GET_TRACEBACK.value)
def get_traceback(batch_id):
    """
    get traceback record by id.
    - input: a batch id.
    - output: if the traceback record can be found, return the traceback in csv format, otherwise, output an error message suggesting not found.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_TRACEBACK.value}")
    log_content, error_code = get_traceback_by_id(int(batch_id))
    if error_code:
        emit(DASHBOARD_EVENT.GET_TRACEBACK.value, error_code)
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
    run_info, error_code = get_run_info_by_id(int(batch_id))
    if error_code:
        emit(DASHBOARD_EVENT.GET_RUN_INFO.value, error_code)
    emit(DASHBOARD_EVENT.GET_RUN_INFO.value, run_info)


@socketio.on(DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value)
def get_interaction_sessions(batch_id):
    """
    get interaction job sessions list
    - input: a batch id.
    - output: if the sessions can be found, return a list of session name, otherwise, return an error message sugesting not found.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value}")
    sessions = get_interaction_sessions_by_id(int(batch_id))
    emit(DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value, sessions)


@socketio.on(DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value)
def get_interaction_sessions(batch_id):
    """
    get interaction job sessions list
    - input: a batch id.
    - output: if the sessions can be found, return a list of session name, otherwise, return an error message sugesting not found.
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value}")
    sessions = get_interaction_sessions_by_id(int(batch_id))
    emit(DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value, sessions)


@socketio.on(DASHBOARD_EVENT.GET_INTERACTION_SESSION_LOG.value)
def get_interaction_session_log(id_info_json):
    """
    get interaction job session log specified by the id info
    - input: infomation about id in a json format:
        {
            "batch_id": <batch id>,
            "session_id": <session id>
        }
    - output: if the session log can be found, return the content of session log, otherwise return an error code
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_INTERACTION_SESSION_LOG.value}")
    id_info_obj = json.loads(id_info_json)
    batch_id = id_info_obj["batch_id"]
    session_id = id_info_obj["session_id"]
    print(f"batch id: {batch_id}, session id: {session_id}")

    log, error_code = get_interaction_session_log_by_id(
        int(id_info_obj["batch_id"]), id_info_obj["session_id"]
    )
    if error_code:
        emit(DASHBOARD_EVENT.GET_INTERACTION_SESSION_LOG.value, error_code)
    emit(DASHBOARD_EVENT.GET_INTERACTION_SESSION_LOG.value, log)


@socketio.on(DASHBOARD_EVENT.GET_DATASET_LIST.value)
def get_dataset_list(pipeline):
    """
    get pipeline specific dataset list
    - input: the pipeline name.
    - output: the list of dataset used in the specified pipeline
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_DATASET_LIST.value}")
    sessions = get_dataset_version_list_by_pipeline(pipeline)
    emit(DASHBOARD_EVENT.GET_DATASET_LIST.value, sessions)


@socketio.on(DASHBOARD_EVENT.GET_DATASET.value)
def get_dataset(dataset_name):
    """
    get specific version of dataset
    - input: the name of the dataset.
    - output: if the dataset can be found, return the dataset content, otherwise return an error code
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value}")
    dataset_content, error_code = get_dataset_by_name(dataset_name)
    if error_code:
        emit(DASHBOARD_EVENT.GET_DATASET.value, error_code)
    emit(DASHBOARD_EVENT.GET_DATASET.value, dataset_content)


@socketio.on(DASHBOARD_EVENT.GET_DATASET_INDECIES.value)
def get_dataset_indices(batch_id):
    """
    get run specific dataset indices
    as for each of the run, more data point can be added to the dataset
    the indices specified the start index and the end index of the data points added to the dataset in a given run
    - input: the batch id of the run.
    - output: [start_index, end_index] of the data added to the dataset with the specified run or error code if cannot find the meta.txt
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_DATASET_INDECIES.value}")
    indices, error_code = get_dataset_indices_by_id(batch_id)
    if error_code:
        emit(DASHBOARD_EVENT.GET_DATASET_INDECIES.value, error_code)
    else:
        emit(DASHBOARD_EVENT.GET_DATASET_INDECIES.value, indices)


@socketio.on(DASHBOARD_EVENT.GET_DATASET_LIST.value)
def get_dataset_list(pipeline):
    """
    get pipeline specific dataset list
    - input: the pipeline name.
    - output: the list of dataset used in the specified pipeline
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_DATASET_LIST.value}")
    sessions = get_dataset_version_list_by_pipeline(pipeline)
    emit(DASHBOARD_EVENT.GET_DATASET_LIST.value, sessions)


@socketio.on(DASHBOARD_EVENT.GET_DATASET.value)
def get_dataset(dataset_name):
    """
    get specific version of dataset
    - input: the name of the dataset.
    - output: if the dataset can be found, return the dataset content, otherwise return an error code
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_INTERACTION_SESSIONS.value}")
    dataset_content, error_code = get_dataset_by_name(dataset_name)
    if error_code:
        emit(DASHBOARD_EVENT.GET_DATASET.value, error_code)
    emit(DASHBOARD_EVENT.GET_DATASET.value, dataset_content)


@socketio.on(DASHBOARD_EVENT.GET_DATASET_INDECIES.value)
def get_dataset_indices(batch_id):
    """
    get run specific dataset indices
    as for each of the run, more data point can be added to the dataset
    the indices specified the start index and the end index of the data points added to the dataset in a given run
    - input: the batch id of the run.
    - output: [start_index, end_index] of the data added to the dataset with the specified run or error code if cannot find the meta.txt
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_DATASET_INDECIES.value}")
    indices, error_code = get_dataset_indices_by_id(batch_id)
    if error_code:
        emit(DASHBOARD_EVENT.GET_DATASET_INDECIES.value, error_code)
    else:
        emit(DASHBOARD_EVENT.GET_DATASET_INDECIES.value, indices)


@socketio.on(DASHBOARD_EVENT.GET_MODEL_KEYS.value)
def get_model_keys(batch_id):
    """
    get all keys for a model related to a run (specified by the batch_id)
    - input: the batch id of the run.
    - output: the keys for the model if the model exists, otherwise error code
    """
    print(f"Request received: {DASHBOARD_EVENT.GET_MODEL_KEYS.value}")
    model, error_code = get_model_by_id(batch_id)
    if error_code:
        emit(DASHBOARD_EVENT.GET_MODEL_KEYS.value, error_code)
    else:
        emit(DASHBOARD_EVENT.GET_MODEL_KEYS.value, get_keys(model))


@socketio.on(DASHBOARD_EVENT.GET_MODEL_VALUE.value)
def get_model_value(batch_id, key):
    """
    get a value for a model related to a run (specified by the batch_id) and the input key
    - input:
        - the batch id of the run.
        - the key for the model, could be any key from the model, or "COMPLETE", indicating getting the complete model dict
    - output: the key and the value specific to the key for the model if the model exists and key is valid, otherwise error code
    """
    print(
        f"Request received: {DASHBOARD_EVENT.GET_MODEL_VALUE.value}, batch_id = {batch_id}, key = {key}"
    )
    model, error_code = get_model_by_id(batch_id)
    if error_code or (key not in get_keys(model) and key != KEY_COMPLETE):
        emit(DASHBOARD_EVENT.GET_MODEL_VALUE.value, error_code)
    elif key == KEY_COMPLETE:
        emit(DASHBOARD_EVENT.GET_MODEL_VALUE.value, get_complete_model(model))
    else:
        # get a specific value
        emit(DASHBOARD_EVENT.GET_MODEL_VALUE.value, [key, get_value_by_key(model, key)])


if __name__ == "__main__":
    socketio.run(app)
