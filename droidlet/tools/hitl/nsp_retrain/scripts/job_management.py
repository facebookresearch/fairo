"""
Job management utils
"""
import boto3
import os
import json

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_ECR_REGION = "us-west-1"
AWS_ECR_REGISTRY_ID = "492338101900"
AWS_ECR_REPO_NAME = "craftassist"

ecr = boto3.client("ecr",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_ECR_REGION,
)

def get_dashboard_version(image_tag: str):
    response = ecr.batch_get_image(
        registryId=AWS_ECR_REGISTRY_ID,
        repositoryName=AWS_ECR_REPO_NAME,
        imageIds=[
            {
                "imageTag": image_tag
            },
        ]
    )
    assert len(response["images"]) == 1
    return response['images'][0]["imageId"]["imageDigest"]

if __name__ == "__main__":
    sha256 = get_dashboard_version("cw_test1")
    print(sha256)
