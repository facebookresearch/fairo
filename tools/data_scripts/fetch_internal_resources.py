import boto3
import subprocess
import os
import sys

def fetch_safety_words():
    """
    Fetch secure s3 resources for internal users and production systems, eg. safety keyword list.

    Currently needs to be called with sys arg passing in output directory.
    """
    try:
        s3 = boto3.client('s3')
        response = s3.head_bucket(Bucket='droidlet-internal')
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print("Authenticated user, fetching safety words list.")
            path_to_root = os.path.realpath(sys.argv[1])
            subprocess.run(
                "aws s3 cp s3://droidlet-internal/safety.txt {}".format(path_to_root), shell=True
            )
    except Exception as e:
        print(e)

if __name__ == "__main__":
    fetch_safety_words()