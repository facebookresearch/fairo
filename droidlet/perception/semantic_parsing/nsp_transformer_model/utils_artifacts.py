"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import subprocess
from subprocess import Popen, PIPE


def compute_checksum_for_model(root_dir=None, agent=None, model_name=None):
    """
    Computes checksum for a given local artifact model and writes it to default_checksum directory
    to help track the hash.
    Args:
        root_dir: The root folder of the project
        agent: Name of agent.
        model_name: name of model (nlu or perception)
    """
    if not root_dir:
        print("Root directory not specified, defaulting to the relative upper four layers")
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../")

    if not agent:
        print("Agent name not specified, defaulting to craftassist agent")
        agent = "craftassist"

    if not model_name:
        print("Model name not given, defaulting to nlu model")
        model_name = "nlu"

    print("Now computing hashes ...")
    # construct the path
    compute_shasum_script_path = os.path.join(
        root_dir, "droidlet/tools/artifact_scripts/checksum_fn.sh"
    )
    checksum_name = ""
    artifact_folder_name = ""
    if model_name == "nlu":
        # compute for NLU model
        artifact_folder_name = "models/" + model_name + "/"
        checksum_name = "nlu.txt"
    elif model_name == "perception":
        # perception models
        artifact_folder_name = "models/" + model_name + "/" + agent
        if agent == "locobot":
            checksum_name = "locobot_perception.txt"
        elif agent == "craftassist":
            checksum_name = "craftassist_perception.txt"

    artifact_path = os.path.join(root_dir, "droidlet/artifacts", artifact_folder_name)
    checksum_write_path = os.path.join(
        root_dir, "droidlet/tools/artifact_scripts/tracked_checksums/" + checksum_name
    )
    result = subprocess.check_output(
        [compute_shasum_script_path, artifact_path, checksum_write_path], text=True
    )
    print(result)

    # read the hash
    with open(checksum_write_path, "r") as f:
        checksum = f.read().strip()
    print("CHECKSUM: %r" % checksum)

    # write the checksum.txt into droidlet/artifacts/model/nlu folder
    if model_name != "nlu":
        model_path = "perception/" + agent
    else:
        model_path = model_name
    with open(
        os.path.join(root_dir, "droidlet/artifacts/models", model_path, "checksum.txt"), "w"
    ) as f:
        f.write(checksum + "\n")

    return checksum


def tar_and_upload(root_dir=None, checksum=None, agent=None, model_name=None):
    if not root_dir:
        print("Root directory not specified, defaulting to the relative upper four layers")
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../")

    if not checksum:
        print("Load checksum from tracked folder")
        if model_name == "nlu":
            checksum_name = "nlu.txt"
        elif model_name == "perception":
            # perception models
            if agent == "locobot":
                checksum_name = "locobot_perception.txt"
            elif agent == "craftassist":
                checksum_name = "craftassist_perception.txt"
        with open(
            os.path.join(
                root_dir, "droidlet/tools/artifact_scripts/tracked_checksums/", checksum_name
            ),
            "r",
        ) as f:
            checksum = f.read().strip()

    if not agent:
        print("Agent name not specified, defaulting to craftassist agent")
        agent = "craftassist"

    if not model_name:
        print("Model name not given, defaulting to nlu model")
        model_name = "nlu"

    # construct the path
    artifact_path_name = os.path.join(root_dir, "droidlet/artifacts/models")
    artifact_path_name = artifact_path_name + "/" + model_name
    artifact_name = "models_" + model_name
    if model_name != "nlu":
        # agent specific models
        artifact_path_name = artifact_path_name + "/" + agent
        artifact_name = artifact_name + "_" + agent
    print(artifact_name, artifact_path_name)
    print("Now making the tar file...")
    process = Popen(
        [
            "tar",
            "-czvf",
            artifact_name + "_" + checksum + ".tar.gz",
            '--exclude="*/\.*"',
            '--exclude="*checksum*"',
            artifact_path_name,
        ],
        stdout=PIPE,
        stderr=PIPE,
    )

    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))
    print("Now uploading ...")
    process = Popen(
        ["aws", "s3", "cp", artifact_name + "_" + checksum + ".tar.gz", "s3://craftassist/pubr/"],
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))
