#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script checks if models and datasets are up to date, and downloads default
# assets (specified in `tool/data_scripts/default_checksums`) if they are stale.
import os
import glob
import sys
import subprocess
from fetch_internal_resources import fetch_safety_words_file

ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
print("Rootdir : %r" % ROOTDIR)


def try_download_artifacts(agent=None):
    # Optionally fetch secure resources for internal users and prod systems
    safety_file_path = os.path.join(ROOTDIR, 'droidlet/documents/internal/safety.txt')
    fetch_safety_words_file(safety_file_path)

    if not agent:
        print("Agent name not specified, defaulting to craftassist")
        agent = "craftassist"

    agent_path = os.path.join(ROOTDIR, 'agents/'+agent)
    print("Agent path: %r" % (agent_path))

    # in case directories don't even exist, create them
    os.makedirs(os.path.join(agent_path, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(agent_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(agent_path, 'models/nlu'), exist_ok=True)
    os.makedirs(os.path.join(agent_path, 'models/perception'), exist_ok=True)

    # Comparing hashes for local directories
    # Default models and datasets shared by all agents
    # Remove existing checksum files so that they can be re-calculated
    fileList = glob.glob(os.path.join(agent_path, 'models/*checksum.txt'), recursive=True)
    for file in fileList:
        print("deleting file :%r" % file)
        os.remove(file)
    dataset_checksum = os.path.join(agent_path, 'datasets/checksum.txt')
    if glob.glob(dataset_checksum):
        os.remove(dataset_checksum)

    compute_shasum_script_path = os.path.join(ROOTDIR, 'tools/data_scripts/checksum_fn.sh')

    artifact_path = os.path.join(agent_path, 'models/nlu')
    checksum_write_path = os.path.join(agent_path, 'models/nlu_checksum.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)
    compare_checksum_try_download(agent, checksum_write_path, "nlu")

    artifact_path = os.path.join(agent_path, 'models/perception')
    checksum_write_path = os.path.join(agent_path, 'models/perception_checksum.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)
    compare_checksum_try_download(agent, checksum_write_path, "perception")

    artifact_path = os.path.join(agent_path, 'datasets')
    checksum_write_path = os.path.join(agent_path, 'datasets/checksum.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)
    compare_checksum_try_download(agent, checksum_write_path, "datasets")


def compare_checksum_try_download(agent=None, local_checksum_file=None, artifact_name=None):
    local_checksum, latest_checksum = None, None
    with open(local_checksum_file) as f:
        local_checksum = f.read().strip()

    if artifact_name == "perception":
        artifact_name = agent + "_" + artifact_name

    latest_checksum_file = os.path.join(ROOTDIR, 'tools/data_scripts/default_checksums/'+artifact_name+'.txt')
    with open(latest_checksum_file) as f:
        latest_checksum = f.read().strip()

    print("Comparing %r checksums" % artifact_name)
    print("Local checksum: %r" % local_checksum)
    print("Latest checksum on remote: %r" % latest_checksum)
    if local_checksum == latest_checksum:
        print("Local checksum for %r is already up to date" % artifact_name)
    else:
        try_download(agent, artifact_name, latest_checksum)


def try_download(agent=None, artifact_name=None, latest_checksum=None):
    print("*********************************************************************************************")
    print("Local %r directory is out of sync. Downloading latest. Use --dev to disable downloads." % artifact_name)
    print("*********************************************************************************************")
    print("Downloading %r directory for %r agent" % (artifact_name, agent))
    if artifact_name == "datasets":
        script_path = os.path.join(ROOTDIR, 'tools/data_scripts/fetch_aws_datasets.sh')
    else:
        # models
        script_path = os.path.join(ROOTDIR, 'tools/data_scripts/fetch_aws_models.sh')

    print("Downloading using script %r" % script_path)
    # TODO: pass down model_name here, also change in upload method
    # nlu: models_folder
    # ca_perception: craftassist_perception
    # locobot : locobot_perception
    result = subprocess.check_output([script_path, agent, latest_checksum, artifact_name],
                                     text=True)
    print(result)
