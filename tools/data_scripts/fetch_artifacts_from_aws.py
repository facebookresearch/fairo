# Copyright (c) Facebook, Inc. and its affiliates.

import os
from subprocess import Popen, PIPE
import shutil


ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
print("Rootdir : %r" % ROOTDIR)


def fetch_artifact_from_aws(agent, artifact_name, model_name, checksum_file_name, checksum_val):
    """
    Uses the agent name, artifact_name, model_name and checksum to fetch the right tar file from s3
    and overwrites corresponding local directory.

    Args:
        agent: agent name - "locobot" or "craftassist"
        artifact_name: "datasets" or "models"
        model_name: "nlu" or "perception"
        checksum_file_name: name of the file containing checksum
        checksum_val: checksum value
    """
    if not agent:
        agent = "craftassist"
        print("Agent name not specified, defaulting to craftassist")
    agent_path = 'agents/' + agent
    print("Agent path: %r" % (agent_path))

    if not checksum_val:
        # if not from command line, read from given file.
        checksum_file_path = os.path.join(ROOTDIR, 'tools/data_scripts/default_checksums/' + checksum_file_name)
        print("Downloading datasets folder with default checksum from file: %r" % checksum_file_path)
        with open(checksum_file_path) as f:
            checksum_val = f.read().strip()
    print("CHECKSUM: %r" % checksum_val)

    artifact_path = os.path.join(agent_path, artifact_name)
    original_artifact_name = artifact_name
    if artifact_name == "models":
        if not model_name:
            model_name = "nlu"
            print("Model type not specified, defaulting to NLU model.")
        artifact_path = artifact_path + "/" + model_name
        artifact_name = artifact_name + '_' + model_name

    file_name = agent + "_" + artifact_name + "_folder_" + checksum_val + ".tar.gz"

    """Get tar file from s3 using : agent name, artifact name and checksum combination as unique identifier"""
    os.chdir(ROOTDIR)
    print("====== Downloading  http://craftassist.s3-us-west-2.amazonaws.com/pubr/" + file_name + " to " \
          + ROOTDIR + file_name + " ======")

    process = Popen(
        [
            'curl',
            'http://craftassist.s3-us-west-2.amazonaws.com/pubr/' + file_name,
            '-o',
            file_name
        ],
        stdout=PIPE,
        stderr=PIPE
    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))

    """Now update the local directory and with untar'd file contents"""
    if os.path.isdir(artifact_path):
        print("Overwriting the directory: %r"% artifact_path)
        shutil.rmtree(artifact_path, ignore_errors=True) # force delete if directory has content in it
    mode = 0o777
    os.mkdir(artifact_path, mode)
    write_path = os.path.join('agents/', agent, original_artifact_name)
    print(write_path)
    process = Popen(
        [
            'tar',
            '-xzvf',
            file_name, '-C',
            write_path,
            '--strip-components',
            '1'
        ],
        stdout=PIPE,
        stderr=PIPE
    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))


def fetch_datasets_from_aws(agent=None, checksum_val=None):
    fetch_artifact_from_aws(agent=agent,
                            artifact_name="datasets",
                            model_name=None,
                            checksum_file_name="datasets.txt",
                            checksum_val=checksum_val)


def fetch_models_from_aws(agent=None, model_name=None, checksum_val=None):
    if not model_name:
        model_name = "nlu"
        print("Model type not specified, defaulting to NLU model.")
    else:
        print("Fetching the model: %r" % model_name)
    # assign checksum file name
    if model_name == "nlu":
        checksum_file = "nlu.txt"
    else:
        checksum_file = agent + "_perception.txt"

    fetch_artifact_from_aws(agent=agent,
                            artifact_name="models",
                            model_name=model_name,
                            checksum_file_name=checksum_file,
                            checksum_val=checksum_val
                            )