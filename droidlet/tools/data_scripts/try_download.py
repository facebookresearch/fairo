# Copyright (c) Facebook, Inc. and its affiliates.
"""
This script checks if models and datasets are up to date, and downloads default
assets (specified in `tool/data_scripts/default_checksums`) if they are stale.
"""
import os
import glob
import subprocess
from droidlet.tools.data_scripts.fetch_internal_resources import fetch_safety_words_file
from droidlet.tools.data_scripts.fetch_artifacts_from_aws import fetch_models_from_aws, \
    fetch_datasets_from_aws, fetch_test_assets_from_aws

ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
print("Rootdir : %r" % ROOTDIR)


def try_download_artifacts(agent=None, test_mode=False):
    """
    Tries to download artifacts if they are out of date.
    """
    # Optionally fetch secure resources for internal users and prod systems
    safety_file_path = os.path.join(ROOTDIR, 'droidlet/documents/internal/safety.txt')
    fetch_safety_words_file(safety_file_path)

    if not agent:
        print("Agent name not specified, defaulting to craftassist")
        agent = "craftassist"

    artifact_path_original = os.path.join(ROOTDIR, 'droidlet/artifacts')
    artifact_path = artifact_path_original
    # in case directories don't exist, create them
    os.makedirs(os.path.join(artifact_path, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, 'models/nlu'), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, 'models/perception'), exist_ok=True)
    os.makedirs(os.path.join(artifact_path, 'models/perception', agent), exist_ok=True)
    if test_mode:
        # Download test artifacts for Locobot tests
        os.makedirs(os.path.join(ROOTDIR, 'droidlet/perception/robot/tests/test_assets/'), exist_ok=True)

    # Remove existing checksum files so that they can be re-calculated
    fileList = [os.path.join(artifact_path, 'models/nlu/nlu_checksum.txt'),
                os.path.join(artifact_path, 'models/perception', agent, 'perception_checksum.txt'),
                os.path.join(artifact_path, 'datasets/checksum.txt')
                ]
    for file in fileList:
        if glob.glob(file):
            print("deleting previous checksum file :%r" % file)
            os.remove(file)

    compute_shasum_script_path = os.path.join(ROOTDIR, 'droidlet/tools/data_scripts/checksum_fn.sh')

    # Compute local checksum for nlu directory and try download if different from remote.

    artifact_path = os.path.join(artifact_path_original, 'models/nlu')
    checksum_write_path = os.path.join(artifact_path, 'nlu_checksum.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)
    compare_checksum_try_download(agent, checksum_write_path, "nlu")

    # Compute and attempt download for perception model
    artifact_path = os.path.join(artifact_path_original, 'models/perception', agent)
    checksum_write_path = os.path.join(artifact_path, 'perception_checksum.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)
    compare_checksum_try_download(agent, checksum_write_path, "perception")

    # Compute and attempt download for datasets
    artifact_path = os.path.join(artifact_path_original, 'datasets')
    checksum_write_path = os.path.join(artifact_path, 'checksum.txt')
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_write_path],
                                     text=True)
    print(result)
    compare_checksum_try_download(agent, checksum_write_path, "datasets")

    if test_mode:
        print("Now downloading test assets...")
        fetch_test_assets_from_aws(agent=agent)


def compare_checksum_try_download(agent=None, local_checksum_file=None, artifact_name=None):
    """
    This function compares local checksum with checksum tracked on main and forces local
    download if the checksums differ.

    Agent:
        agent: Name of agent - "locobot" or "craftassist"
        local_checksum_file: path to local checksum file inside agent folder.
        artifact_name: model name ("perception" or "nlu") or "datasets"
    """
    local_checksum, latest_checksum = None, None
    with open(local_checksum_file) as f:
        local_checksum = f.read().strip()

    artifact_name_file = artifact_name
    if artifact_name == "perception":
        # We are tracking locobot_perception and craftassist_perception on remote
        artifact_name_file = agent + "_" + artifact_name

    latest_checksum_file = os.path.join(ROOTDIR,
                                        'droidlet/tools/data_scripts/default_checksums/' + artifact_name_file + '.txt')
    with open(latest_checksum_file) as f:
        latest_checksum = f.read().strip()

    print("Comparing %r checksums for agent: %r" % (artifact_name, agent))
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
        fetch_datasets_from_aws(agent=agent, checksum_val=latest_checksum)
    else:
        # models
        fetch_models_from_aws(agent=agent, model_name=artifact_name, checksum_val=latest_checksum)


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
        "--test_mode",
        action="store_true"
    )
    args = parser.parse_args()
    try_download_artifacts(agent=args.agent_name, test_mode=args.test_mode)