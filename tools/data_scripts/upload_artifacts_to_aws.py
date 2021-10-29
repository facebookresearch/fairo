# Copyright (c) Facebook, Inc. and its affiliates.

"""This script creates a tar and hash of the artifacts directory.
If uploading files to S3 through console UI, go to the web interface at:
https://s3.console.aws.amazon.com/s3/buckets/craftassist?region=us-west-2&prefix=pubr/&showversions=false
and upload the tar.gz file.
"""
import os
import subprocess
from subprocess import Popen, PIPE


ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
print("Rootdir : %r" % ROOTDIR)


def compute_checksum_tar_and_upload(agent, artifact_name, model_name=None):
    if not agent:
        agent = "craftassist"
        print("Agent name not specified, defaulting to craftassist")

    agent_path = os.path.join(ROOTDIR, 'agents/' + agent)
    print("Agent path: %r" % (agent_path))

    """Compute checksum for local artifact folder (to check in to tar file)"""
    checksum_name = 'checksum.txt'
    artifact_path_name = artifact_name + "/"
    artifact_path = os.path.join(agent_path, artifact_name)
    compute_shasum_script_path = os.path.join(ROOTDIR, 'tools/data_scripts/checksum_fn.sh')

    if artifact_name == "models":
        if not model_name:
            model_name = "nlu"
            print("Model type not specified, defaulting to NLU model.")
        checksum_name = model_name + '_' + checksum_name
        artifact_path_name = artifact_path_name + model_name + "/"
        artifact_path = artifact_path + "/" + model_name
        artifact_name = artifact_name + '_' + model_name

    checksum_path = os.path.join(artifact_path, checksum_name)
    os.chdir(agent_path)
    result = subprocess.check_output([compute_shasum_script_path, artifact_path, checksum_path],
                                     text=True)
    print(result)

    with open(checksum_path) as f:
        checksum = f.read().strip()
    print("CHECKSUM: %r" % checksum)

    """Tar and upload the local artifact folder"""
    print("Now making the tar file...")
    process = Popen(['tar', '-czvf',
                                agent + '_' + artifact_name + '_folder_' + checksum + '.tar.gz',
                                '--exclude="*/\.*"', '--exclude="*checksum*"', artifact_path_name],
                               stdout=PIPE,
                               stderr=PIPE
                               )
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))
    print("Now uploading ...")
    process = Popen(['aws', 's3', 'cp',
                agent + '_' + artifact_name + '_folder_' + checksum + '.tar.gz',
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

