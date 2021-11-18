# Copyright (c) Facebook, Inc. and its affiliates.

import os
from subprocess import Popen, PIPE
import shutil

ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
print("Rootdir : %r" % ROOTDIR)


def fetch_test_assets_from_aws(agent=None):
    assert agent == "locobot"
    file_name = 'locobot_perception_test_assets.tar.gz'
    aws_asset_file = 'https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_test_assets.tar.gz'
    os.chdir(ROOTDIR)
    print("====== Downloading " + aws_asset_file + " to " + ROOTDIR + file_name + " ======")

    process = Popen(
        [
            'curl',
            'https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_test_assets.tar.gz',
            '-o',
            file_name
        ],
        stdout=PIPE,
        stderr=PIPE
    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))

    test_artifact_path = os.path.join(ROOTDIR, 'droidlet/perception/robot/tests/test_assets/')
    """Now update the local directory and with untar'd file contents"""
    if os.path.isdir(test_artifact_path):
        print("Overwriting the directory: %r" % test_artifact_path)
        shutil.rmtree(test_artifact_path, ignore_errors=True)  # force delete if directory has content in it
    mode = 0o777
    os.mkdir(test_artifact_path, mode)

    print("Writing to : %r" % test_artifact_path)
    process = Popen(
        [
            'tar',
            '-xzvf',
            file_name, '-C',
            test_artifact_path,
            '--strip-components',
            '1'
        ],
        stdout=PIPE,
        stderr=PIPE
    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))


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

    if not checksum_val:
        # if not from command line, read from given file.
        checksum_file_path = os.path.join(ROOTDIR,
                                          'droidlet/tools/data_scripts/default_checksums/' + checksum_file_name)
        print("Downloading datasets folder with default checksum from file: %r" % checksum_file_path)
        with open(checksum_file_path) as f:
            checksum_val = f.read().strip()
    print("CHECKSUM: %r" % checksum_val)

    artifact_path = os.path.join('droidlet/artifacts/', artifact_name)
    write_path = artifact_path
    if artifact_name == "models":
        if not model_name:
            model_name = "nlu"
            print("Model type not specified, defaulting to NLU model.")
        artifact_path = artifact_path + "/" + model_name
        artifact_name = artifact_name + '_' + model_name
        if not os.path.isdir(artifact_path):
            mode = 0o777
            os.mkdir(artifact_path, mode)
        if model_name != "nlu":
            artifact_path = artifact_path + "/" + agent
            artifact_name = artifact_name + "_" + agent

    file_name = artifact_name + "_" + checksum_val + ".tar.gz"
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
        print("Overwriting the directory: %r" % artifact_path)
        shutil.rmtree(artifact_path, ignore_errors=True)  # force delete if directory has content in it
    mode = 0o777
    os.mkdir(artifact_path, mode)


    print("Writing to : %r" % write_path)
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
        print("Model type not specified, defaulting to NLU model for agent: %r." % agent)
    else:
        print("Fetching the model: %r for agent :%r " % (model_name, agent))
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in agent name to download artifacts for.")
    parser.add_argument(
        "--agent_name",
        help="Name of the agent",
        type=str,
        default="craftassist",
    )
    parser.add_argument(
        "--artifact_name",
        help="Name of artifact",
        type=str,
        default="models"
    )
    parser.add_argument(
        "--model_name",
        help="Name of model when artifact name is models",
        type=str,
        default="nlu"
    )
    parser.add_argument(
        "--checksum_file",
        help="name of checksum file",
        type=str,
        default="nlu.txt"
    )
    args = parser.parse_args()
    fetch_artifact_from_aws(agent=args.agent_name,
                            artifact_name=args.artifact_name,
                            model_name=args.model_name,
                            checksum_file_name=args.checksum_file,
                            checksum_val=None
                            )

