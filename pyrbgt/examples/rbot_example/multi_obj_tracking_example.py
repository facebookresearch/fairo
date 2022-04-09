from pyrbgt import RBGTTracker
from rbot_dataset_handle import RBOTDatasetHandle
import time
import os

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# TODO: Make this as a command line arg using argparse
# Modify this to actual path of RBOT_dataset.
dataset_path = "~/data/RBOT_dataset/"

config = dotdict(
    visualize=True,
    evaluate=False,
    evaluate_dataset_path=dataset_path,
    models=[
        dotdict(
            model="ape",
            name="ape",
            path=os.path.join(dataset_path, "ape"),
            model_filename="ape.obj",
            unit_in_meter=0.001,
        ),
        dotdict(
            model="squirrel_small",
            name="squirrel_small",
            path=dataset_path,
            model_filename="squirrel_small.obj",
            unit_in_meter=0.001,
        ),
    ],
)

tracker = RBGTTracker(config)
image_handle = RBOTDatasetHandle(dataset_path, "ape", "d_occlusion")
tracker.track(image_handle)
for _ in range(100):
    print(tracker.output())
    time.sleep(0.025)
