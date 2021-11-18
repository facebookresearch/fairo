# Copyright (c) Facebook, Inc. and its affiliates.

"""
This script computes hashes for the given local artifact directories and saves them to data_scripts/default_checksums/
"""
import os
import subprocess


def compute_checksum_for_directory(
        agent=None,
        artifact_type=None,
        model_name=None):
    """
    Computes checksum for a given local artifact directory and writes it to default_checksum directory
    to help track the hash.

    Args:
        agent: Name of agent.
        artifact_type: datasets or models artifact
        model_name: name of model (nlu or perception)
    """
    ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
    print("Rootdir : %r" % ROOTDIR)

    if not agent:
        print("Agent name not specified, defaulting to craftassist agent")
        agent = "craftassist"

    if not artifact_type:
        print("Artifact not specified, defaulting to models/")
        artifact_type = "models"

    if artifact_type == "models" and not model_name:
        print("Model name not given, defaulting to nlu model")
        model_name = "nlu"

    print("Now computing hashes ...")
    compute_shasum_script_path = os.path.join(ROOTDIR, 'droidlet/tools/data_scripts/checksum_fn.sh')
    checksum_name = ''
    artifact_folder_name = ''
    if artifact_type == "models":
        artifact_folder_name = 'models/' + model_name + "/" + agent
        if model_name == "nlu":
            # compute for NLU model
            checksum_name = 'nlu.txt'
        elif model_name == "perception":
            # perception models
            if agent == "locobot":
                checksum_name = 'locobot_perception.txt'
            elif agent == "craftassist":
                checksum_name = 'craftassist_perception.txt'
    else:
        # datasets
        artifact_folder_name = 'datasets/'
        checksum_name = 'datasets.txt'

    artifact_path = os.path.join(ROOTDIR, 'droidlet/artifacts', artifact_folder_name)
    checksum_write_path = os.path.join(ROOTDIR, 'droidlet/tools/data_scripts/default_checksums/' + checksum_name)
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)