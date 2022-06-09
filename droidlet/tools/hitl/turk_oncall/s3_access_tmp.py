import os
import boto3
import botocore
import tarfile

ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30
INTERACTION_LISTENER_POLL_TIME = 30
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

batch_id = 20220224132033
batch_prefix = f"{batch_id}/interaction/"
batch_bucket = s3.Bucket(S3_BUCKET_NAME)

tmp_dir = os.path.join(HITL_TMP_DIR, "tmp")

def data_gen_draft(file_path, folder_path):
    # unzip
    file = tarfile.open(file_path)
    file.extractall(folder_path)
    file.close()

for obj in batch_bucket.objects.filter(Prefix=batch_prefix):
    dest = os.path.join(tmp_dir, obj.key)
    folder_path = dest[:dest.rindex('/')]
    os.makedirs(folder_path, exist_ok=True)
    print('retreiving %s from s3' % obj.key)

    try:
        s3.Bucket(S3_BUCKET_NAME).download_file(obj.key, dest)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    # pass to data generator
    data_gen_draft(dest, folder_path)









    
