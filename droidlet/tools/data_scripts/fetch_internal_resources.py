import boto3
import subprocess
import os

def fetch_safety_words_file(file_path):
    """
    Fetch secure s3 resources for internal users and production systems, eg. safety keyword list.

    Currently needs to be called with sys arg passing in output directory.
    """
    try:
        s3 = boto3.client('s3')
        response = s3.head_bucket(Bucket='droidlet-internal')
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print("Authenticated user, fetching safety words list.")
            path_to_root = os.path.realpath(file_path)
            return subprocess.run(
                "aws s3 cp s3://droidlet-internal/safety.txt {}".format(path_to_root), shell=True
            )
    except Exception as e:
        print(e)
        pass
    # If awscli is not setup, eg. in one time use containers, read access tokens from environment   
    try:
        s3 = boto3.client('s3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        print("Set up boto3 s3 client, attempting to download internal resources.")
        s3.download_file(
            'droidlet-internal',
            'safety.txt',
            os.path.realpath(file_path)
        )
    except Exception as e:
        print(e)
