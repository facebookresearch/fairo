"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys
import json
import subprocess
import time

EXPECTED_KEYS = ["blocks", "inst_seg_tags", "avatarInfo"]


def launch_scene_viewer(scene_list: dict):
    """
    Launches the scene viewer located in the same folder to show the scene passed in as an argument.
    Saves the scene as 'scene_list.json' in the local dir
    Deletes the temporary scene file on exit
    Scene passed must be formatted correctly (see above expected keys)
    """

    # Only checking the first scene; assuming they are all formatted the same
    assert all(
        [x in scene_list[0].keys() for x in EXPECTED_KEYS]
    ), "Scene not formatted correctly!"

    # Set the working directory to be this one, in case this function was imported
    # Have to do this to have access to supporting html and js files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Save the scene list locally
    with open("scene_list.json", "w") as f:
        json.dump(scene_list, f)

    # Instruct user
    print("Launching Servez...")
    print("\033[91mWhen you open the browser, select `MODEL_OUTPUT_VIEWER.html`\033[0m")
    time.sleep(3)

    # Launch servez
    servez_cmd = "servez --no-index"
    try:
        subprocess.Popen(servez_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    except:
        print("Are you sure you have Servez installed?  Run `npm install servez -g`")
        raise

    print("\033[93mDone? Press CTRL-C to kill servez and delete the scene file\n\033[0m")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        os.remove("scene_list.json")
        raise


if __name__ == "__main__":
    # Generate example scenes
    scene_save_path = os.path.join(os.getcwd(), "scene_list.json")
    scene_gen_path = os.path.join(
        os.getcwd(), "../../../../../lowlevel/minecraft/small_scenes_with_shapes.py"
    )
    scene_gen_cmd = (
        "python3 "
        + scene_gen_path
        + " --SL="
        + str(17)
        + " --H="
        + str(13)
        + " --GROUND_DEPTH="
        + str(5)
        + " --MAX_NUM_SHAPES="
        + str(4)
        + " --NUM_SCENES="
        + str(3)
        + " --MAX_NUM_GROUND_HOLES="
        + str(2)
        + " --save_data_path="
        + scene_save_path
        + " --iglu_scenes="
        + os.environ["IGLU_SCENE_PATH"]
    )
    print("Starting scene generation script")
    scene_gen = subprocess.Popen(
        scene_gen_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True
    )
    while scene_gen.poll() is None:
        time.sleep(1)

    with open("scene_list.json", "r") as f:
        js = json.load(f)
    launch_scene_viewer(js)
