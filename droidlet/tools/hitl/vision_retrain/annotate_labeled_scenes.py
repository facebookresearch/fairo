import os
import json
import re
import argparse
from datetime import datetime

from droidlet.tools.hitl.task_runner import TaskRunner
from droidlet.tools.hitl.utils.hitl_utils import generate_batch_id
from droidlet.tools.hitl.vision_retrain.vision_annotation_jobs import VisionAnnotationJob

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)


def main(batch_ids: list, timeout: int):
    scene_list = []
    annotation_batch_id = generate_batch_id()

    # For each batch ID, scrape the local .hitl folder for labeled, unannotated scenes
    print("Scraping batch_id directories for unannotated scenes")
    for batch_id in batch_ids:
        anno_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/annotated_scenes")
        potential_files = [f for f in os.listdir(anno_dir) if re.match("\d+\_unannotated.json", f)]
        if len(potential_files) == 0:
            print(f"WARNING did not find a match for batch_id {batch_id}")
            print("Check file name and maybe run `recover_unannotated_scenes.py`")
        else:
            if len(potential_files) > 1:
                print(
                    f"WARNING found more than one match for batch_id {batch_id} ; Only using the first."
                )
            unannotated_filename = potential_files[0]
            unannotated_filepath = os.path.join(anno_dir, unannotated_filename)
            with open(unannotated_filepath, "r") as f:
                js = json.load(f)
            print(f"Merging {len(js)} unannotated scenes from batch_id {batch_id}")
            scene_list.extend(js)

            # Mark the old unannotated file as reannotated
            reannotated_filename = (
                unannotated_filename[:-5] + f"_reannotated{annotation_batch_id}.json"
            )
            reannotated_filepath = os.path.join(anno_dir, reannotated_filename)
            os.rename(unannotated_filepath, reannotated_filepath)
            print(f"Renamed {unannotated_filename} as {reannotated_filename} for posterity")

    assert len(scene_list) > 0, "No unannotated scenes found in the given batch_id folders"

    # Launch a new annotation job
    print(
        f"Scene list compiled, contains {len(scene_list)} scenes. Running annotation job: {annotation_batch_id}."
    )
    runner = TaskRunner()

    aj = VisionAnnotationJob(
        batch_id=annotation_batch_id,
        timestamp=int(datetime.utcnow().timestamp()),
        scenes=scene_list,
        timeout=timeout,
    )
    runner.register_data_generators([aj])
    runner.run()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=120)
    opts = parser.parse_args()
    timeout = opts.timeout

    batch_ids = []
    while True:
        batch_id = input("Add a batch_id to reference, or enter to continue: ")
        if not batch_id:
            break
        batch_ids.append(batch_id)

    if len(batch_ids) == 0:
        print("You must enter at least one batch_id to continue")
    else:
        print(f"Combining labeled scenes from the following batch_ids: {batch_ids}")
        main(batch_ids, timeout)
