# Copyright (c) Facebook, Inc. and its affiliates.

"""This script creates a tar and hash of the artifacts directory and uploads it to S3.
If uploading files to S3 through console UI, go to the web interface at:
https://s3.console.aws.amazon.com/s3/buckets/craftassist?region=us-west-2&prefix=pubr/&showversions=false
and upload the tar.gz file.
"""
import os
import subprocess
from subprocess import Popen, PIPE

from droidlet.tools.artifact_scripts.compute_checksum import compute_checksum_for_directory


ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
print("Rootdir : %r" % ROOTDIR)


def tar_and_upload(agent, artifact_name, model_name=None, checksum=None):
    """
    Tar all files for the artifact and upload it to AWS.
    """
    if not agent:
        print("Agent name not specified, defaulting to craftassist agent")
        agent = "craftassist"

    # construct the path
    artifact_path_name = artifact_name + "/"
    if artifact_name == "models":
        if not model_name:
            model_name = "nlu"
            print("Model type not specified, defaulting to NLU model.")
        artifact_path_name = artifact_path_name + model_name
        artifact_name = artifact_name + "_" + model_name
        if model_name != "nlu":
            # agent specific models
            artifact_path_name = artifact_path_name + "/" + agent
            artifact_name = artifact_name + "_" + agent
        print(artifact_name, artifact_path_name)

    # Change the directory to artifacts
    os.chdir(os.path.join(ROOTDIR, "droidlet/artifacts/"))

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


def upload_artifacts_to_aws(agent, artifact_name, model_name=None):
    if not agent:
        agent = "craftassist"
        print("Agent name not specified, defaulting to craftassist")

    """Compute checksum for local artifact folder (to check in to tar file)"""
    checksum_name = "checksum.txt"
    artifact_path = os.path.join(ROOTDIR, "droidlet/artifacts/", artifact_name)
    print("Artifact path: %r" % artifact_path)
    if artifact_name == "models":
        if not model_name:
            model_name = "nlu"
            print("Model type not specified, defaulting to NLU model.")
        checksum_name = model_name + "_" + checksum_name
        artifact_path = artifact_path + "/" + model_name
        if model_name != "nlu":
            artifact_path = artifact_path + "/" + agent
    checksum_path = os.path.join(artifact_path, checksum_name)

    compute_checksum_for_directory(agent, artifact_name, model_name, checksum_path)

    # Read hash from file
    with open(checksum_path) as f:
        checksum = f.read().strip()
    print("CHECKSUM: %r" % checksum)

    # Tar the folder and upload to AWS
    tar_and_upload(agent, artifact_name, model_name, checksum)


def upload_agent_datasets(agent=None):
    upload_artifacts_to_aws(agent=agent, artifact_name="datasets")


def upload_agent_models(agent=None, model_name=None):
    upload_artifacts_to_aws(agent=agent, artifact_name="models", model_name=model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in agent name to download artifacts for.")
    parser.add_argument(
        "--agent_name",
        help="Name of the agent",
        type=str,
        default="craftassist",
    )
    parser.add_argument("--artifact_name", help="Name of artifact", type=str, default="models")
    parser.add_argument(
        "--model_name", help="Name of model when artifact name is models", type=str, default="nlu"
    )
    args = parser.parse_args()
    upload_artifacts_to_aws(
        agent=args.agent_name, artifact_name=args.artifact_name, model_name=args.model_name
    )
