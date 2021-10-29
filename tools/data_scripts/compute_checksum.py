# Copyright (c) Facebook, Inc. and its affiliates.

"""
This script computes hashes for local directories and saves them to data_scripts/default_checksums/
# ./compute_checksum.sh craftassist # hash for semantic parser models
# ./compute_checksum.sh craftassist datasets # hash for datasets folder
# ./compute_checksum.sh locobot # hash for locobot models

"""
import os
import subprocess


def compute_checksum_for_directory(agent=None, artifact_type=None, artifact_name=None):
    ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
    print("Rootdir : %r" % ROOTDIR)

    if not agent:
        print("Agent name not specified, defaulting to craftassist")
        agent = "craftassist"

    if not artifact_type:
        print("Artifact not specified, defaulting to models/")
        artifact_type = "models"

    if artifact_type and not artifact_name:
        print("Model name not given, defaulting to nlu model")
        artifact_name = "nlu"

    agent_path = os.path.join(ROOTDIR, 'agents/'+agent)

    print("Now computing hashes ...")
    compute_shasum_script_path = os.path.join(ROOTDIR, 'tools/data_scripts/checksum_fn.sh')
    if artifact_type == "models":
        # TODO: rename the following folder and checksum files
        if artifact_name == "nlu":
            # compute for NLU model
            artifact_path = os.path.join(agent_path, 'models/semantic_parser')
            checksum_write_path = os.path.join(ROOTDIR, 'tools/data_scripts/default_checksums/nsp.txt')
        elif artifact_name == "perception":
            # perception models
            artifact_path = os.path.join(agent_path, 'models/perception')
            if agent == "locobot":
                checksum_write_path = os.path.join(ROOTDIR, 'tools/data_scripts/default_checksums/locobot.txt')
            elif agent == "craftassist":
                checksum_write_path = os.path.join(ROOTDIR, 'tools/data_scripts/default_checksums/craftassist_perception.txt')
    else:
        # datasets
        artifact_path = os.path.join(agent_path, 'datasets/')
        checksum_write_path = os.path.join(ROOTDIR, 'tools/data_scripts/default_checksums/datasets.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)