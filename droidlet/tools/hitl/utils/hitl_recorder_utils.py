"""
Copyright (c) Facebook, Inc. and its affiliates.

Utils that helps to reterive detail infomation about a job 
for recording the job infomation. 
"""
import os
import boto3

# for s3
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

# for ecr
AWS_ECR_ACCESS_KEY_ID = os.environ["AWS_ECR_ACCESS_KEY_ID"]
AWS_ECR_SECRET_ACCESS_KEY = os.environ["AWS_ECR_SECRET_ACCESS_KEY"]
AWS_ECR_REGION = os.environ["AWS_ECR_REGION"] if os.getenv("AWS_ECR_REGION") else "us-west-1"
AWS_ECR_REGISTRY_ID = "492338101900"
AWS_ECR_REPO_NAME = "craftassist"

ecr = boto3.client(
    "ecr",
    aws_access_key_id=AWS_ECR_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_ECR_SECRET_ACCESS_KEY,
    region_name=AWS_ECR_REGION,
)


def get_dashboard_version(image_tag: str):
    """
    get sha256 value of the input image_tag
    """
    response = ecr.batch_get_image(
        registryId=AWS_ECR_REGISTRY_ID,
        repositoryName=AWS_ECR_REPO_NAME,
        imageIds=[
            {"imageTag": image_tag},
        ],
    )
    assert len(response["images"]) == 1
    return response["images"][0]["imageId"]["imageDigest"]


def get_s3_link(batch_id: int):
    """
    get s3 link for the batch with batch_id input
    """
    return f"https://s3.console.aws.amazon.com/s3/buckets/droidlet-hitl?region={AWS_DEFAULT_REGION}&prefix={batch_id}"


if __name__ == "__main__":
    sha256 = get_dashboard_version("cw_test1")
    print(sha256)
