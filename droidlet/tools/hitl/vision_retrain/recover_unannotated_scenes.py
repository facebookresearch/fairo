import os
import json
import re
import argparse
import boto3

from droidlet.lowlevel.minecraft.shape_util import SHAPE_NAMES

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
S3_BUCKET_NAME = "droidlet-hitl"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)


def main(batch_id: int):

    # Pull the completed annotated list from .hitl/batch_id
    print("Retrieving nominally annotated scene list")
    anno_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/annotated_scenes")
    partially_anno_filename = [f for f in os.listdir(anno_dir) if re.match("\d+\.json", f)][0]
    partially_anno_filepath = os.path.join(anno_dir, partially_anno_filename)
    with open(partially_anno_filepath, "r") as f:
        partially_anno_scenes = json.load(f)

    # Filter scenes with invalid inst_seg_tags
    print("Filtering out invalid tags")
    not_done = [
        x
        for x in partially_anno_scenes
        if not x["inst_seg_tags"] or x["inst_seg_tags"][0]["tags"][0] in SHAPE_NAMES
    ]
    clean_anno_scenes = [
        x
        for x in partially_anno_scenes
        if x["inst_seg_tags"] and x["inst_seg_tags"][0]["tags"][0] not in SHAPE_NAMES
    ]
    assert (len(not_done) + len(clean_anno_scenes)) == len(partially_anno_scenes)
    print(f"Found {len(not_done)} scenes that appear to be unannotated")

    # Save the two scene lists locally
    print("Saving filtered scene lists to .hitl directory")
    unannotated_filename = partially_anno_filename[:-5] + "_unannotated.json"
    unannotated_filepath = os.path.join(anno_dir, unannotated_filename)
    with open(unannotated_filepath, "w") as f:
        json.dump(not_done, f)

    clean_filename = partially_anno_filename[:-5] + "_clean.json"
    clean_filepath = os.path.join(anno_dir, clean_filename)
    with open(clean_filepath, "w") as f:
        json.dump(clean_anno_scenes, f)

    # Upload the scene lists to S3
    print(f"Uploading filtered scene lists to S3 under batch_id ({batch_id}) key")
    with open(clean_filepath, "rb") as f:
        s3.upload_fileobj(
            f,
            f"{S3_BUCKET_NAME}",
            f"{batch_id}/annotated_scenes/{clean_filename}",
        )
    with open(unannotated_filepath, "rb") as f:
        s3.upload_fileobj(
            f,
            f"{S3_BUCKET_NAME}",
            f"{batch_id}/annotated_scenes/{unannotated_filename}",
        )

    print(f"Done cleaning scene list from annotation job: {batch_id}")
    print(
        f"{unannotated_filename} can be combined with other labeled scene lists and used to launch a new annotation job"
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=int, help="The batch_id of the job to clean")
    args = parser.parse_args()
    batch_id = args.batch_id

    main(batch_id)
