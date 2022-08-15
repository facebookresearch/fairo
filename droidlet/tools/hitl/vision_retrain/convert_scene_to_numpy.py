"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import json
import argparse
import boto3
import re
import numpy as np
import pickle

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)


def main(batch_id: int, scene_filename: str, save_path: str):
    """
    Takes in the output of an annotation job.
    Outputs the scene list in the format required by the vision model training script.
    """

    # Figure out the scene filename from the batch ID if not given
    if not scene_filename:
        print("No scene file name provided, looking for one in .hitl")
        anno_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/annotated_scenes")
        filenames = [f for f in os.listdir(anno_dir) if re.match("\d+\_clean.json", f)]
        assert len(filenames) > 0, "Scene filename not given or found in local .hitl"
        scene_filename = filenames[0]

    # Download the scene from S3 and load into a json file
    print("Downloading scene list from S3 annotated_scenes")
    try:
        s3.download_file(
            f"{S3_BUCKET_NAME}",
            f"{batch_id}/annotated_scenes/{scene_filename}",
            "scene_list.json",
        )
    except:
        print("Scene file not found on S3 where expected, check that it exists")
        raise

    with open("scene_list.json", "r") as f:
        js = json.load(f)
    assert isinstance(js, list), "JSON scene file appears improperly formatted"

    print("Converting scene list to model training format")
    output_scenes = []
    scenes_complete = 0
    for scene in js:
        output_scene = []

        # Handle potential xyz offset
        xmin, ymin, zmin = np.inf, np.inf, np.inf
        xmax, ymax, zmax = np.NINF, np.NINF, np.NINF
        for block in scene["blocks"]:
            if block[0] < xmin:
                xmin = block[0]
            if block[0] > xmax:
                xmax = block[0]
            if block[1] < ymin:
                ymin = block[1]
            if block[1] > ymax:
                ymax = block[1]
            if block[2] < zmin:
                zmin = block[2]
            if block[2] > zmax:
                zmax = block[2]
        xrange, yrange, zrange = int(xmax - xmin + 1), int(ymax - ymin + 1), int(zmax - zmin + 1)

        # Populate the bid npy array
        npy_blocks = np.zeros([xrange, yrange, zrange], dtype=np.int32)
        for block in scene["blocks"]:
            npy_blocks[block[0] - xmin][block[1] - ymin][block[2] - zmin] = block[3]
        output_scene.append(npy_blocks)

        # Populate the tag index map
        tag_locs = scene["inst_seg_tags"][0]["locs"]
        npy_index_map = np.zeros([xrange, yrange, zrange], dtype=np.int32)
        for loc in tag_locs:
            try:
                npy_index_map[loc[0] - xmin][loc[1] - ymin][loc[2] - zmin] = 1
            except IndexError:
                # If the annotator marked a block outside the scene, ignore it
                continue
        output_scene.append(npy_index_map)

        # Create the list of tags
        assert (
            len(scene["inst_seg_tags"][-1]["tags"]) == 1
        ), "Scene appears to have the wrong number of tags (should be one)"
        # take the rightmost entry, possible that worker submitted a new annotation that was appended
        output_scene.append(["nothing", scene["inst_seg_tags"][-1]["tags"][0]])

        output_scenes.append(output_scene)
        scenes_complete += 1
        if scenes_complete % 50 == 0:
            print(f"{scenes_complete} scenes complete")

    # Pickle, upload, and store locally in .hitl
    print("Scenes successfully converted, storing locally and on S3")
    if not save_path:
        print("No save path provided, using default .hitl/batch_id")
        save_path = os.path.join(HITL_TMP_DIR, f"{batch_id}/model_training_data")
    os.makedirs(save_path, exist_ok=True)
    output_filename = scene_filename[:-5] + "_modeldata.pkl"
    output_filepath = os.path.join(save_path, output_filename)
    with open(output_filepath, "wb") as f:
        pickle.dump(output_scenes, f)
    with open(output_filepath, "rb") as f:
        s3.upload_fileobj(
            f,
            f"{S3_BUCKET_NAME}",
            f"{batch_id}/model_training_data/{output_filename}",
        )

    # Delete the temporary download file
    os.remove("scene_list.json")
    print(f"Scene list conversion for batch_id {batch_id} complete, exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_id", type=int, help="The batch_id of the annotation job to convert"
    )
    parser.add_argument(
        "--scene_filename",
        type=str,
        default="",
        help="The .json filename of the scene to convert, default will search .hitl",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Path to save the reformatted data, default .hitl/batch_id",
    )
    opts = parser.parse_args()

    main(opts.batch_id, opts.scene_filename, opts.save_path)
