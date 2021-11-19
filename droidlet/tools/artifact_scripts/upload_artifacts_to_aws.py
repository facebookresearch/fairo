# Copyright (c) Facebook, Inc. and its affiliates.

"""This script creates a tar and hash of the artifacts directory.
If uploading files to S3 through console UI, go to the web interface at:
https://s3.console.aws.amazon.com/s3/buckets/craftassist?region=us-west-2&prefix=pubr/&showversions=false
and upload the tar.gz file.
"""
import os
import subprocess
from subprocess import Popen, PIPE


ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
print("Rootdir : %r" % ROOTDIR)


def compute_checksum_tar_and_upload(agent, artifact_name, model_name=None):
    if not agent:
        agent = "craftassist"
        print("Agent name not specified, defaulting to craftassist")

    """Compute checksum for local artifact folder (to check in to tar file)"""
    checksum_name = 'checksum.txt'
    artifact_path_name = artifact_name + "/"
    artifact_path = os.path.join(ROOTDIR, "droidlet/artifacts/", artifact_name)
    compute_shasum_script_path = os.path.join(ROOTDIR, 'droidlet/tools/artifact_scripts/checksum_fn.sh')
    print("Artifact path: %r" % artifact_path)
    if artifact_name == "models":
        if not model_name:
            model_name = "nlu"
            print("Model type not specified, defaulting to NLU model.")
        checksum_name = model_name + '_' + checksum_name
        artifact_path_name = artifact_path_name + model_name
        artifact_path = artifact_path + "/" + model_name
        artifact_name = artifact_name + '_' + model_name
        if model_name != "nlu":
            # agent specific models
            artifact_path_name  = artifact_path_name + "/" + agent
            artifact_path = artifact_path  + "/" + agent
            artifact_name = artifact_name + "_" + agent
        print(artifact_name, artifact_path, artifact_path_name)
    checksum_path = os.path.join(artifact_path, checksum_name)
    os.chdir(os.path.join(ROOTDIR, "droidlet/artifacts/"))
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_path],
                                     text=True)
    print(result)

    with open(checksum_path) as f:
        checksum = f.read().strip()
    print("CHECKSUM: %r" % checksum)

    """Tar and upload the local artifact folder"""
    print("Now making the tar file...")
    process = Popen(['tar',
                     '-czvf',
                     artifact_name + "_" + checksum + '.tar.gz',
                     '--exclude="*/\.*"',
                     '--exclude="*checksum*"',
                     artifact_path_name],
                    stdout=PIPE,
                    stderr=PIPE
                    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))
    print("Now uploading ...")
    process = Popen(['aws', 's3', 'cp',
                     artifact_name + "_" + checksum + '.tar.gz',
                     's3://craftassist/pubr/'],
                    stdout=PIPE,
                    stderr=PIPE
                    )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))


def upload_agent_datasets(agent=None):
    compute_checksum_tar_and_upload(agent=agent, artifact_name="datasets")


def upload_agent_models(agent=None, model_name=None):
    compute_checksum_tar_and_upload(agent=agent, artifact_name="models", model_name=model_name)


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
    args = parser.parse_args()
    compute_checksum_tar_and_upload(agent=args.agent_name,
                                    artifact_name=args.artifact_name,
                                    model_name=args.model_name)
