from pyrbgt import RBGTTracker
from rbot_dataset_handle import RBOTDatasetHandle
import time


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = dotdict(
    visualize=True,
    evaluate=False,
    evaluate_dataset_path="/data/RBOT_dataset/",
    models=[
        dotdict(
            model="ape",
            name="ape",
            path="/data/RBOT_dataset/ape/",
            model_filename="ape.obj",
            unit_in_meter=0.001,
        ),
        dotdict(
            model="squirrel_small",
            name="squirrel_small",
            path="/data/RBOT_dataset/",
            model_filename="squirrel_small.obj",
            unit_in_meter=0.001,
        ),
    ],
)

tracker = RBGTTracker(config)
image_handle = RBOTDatasetHandle("/data/RBOT_dataset/", "ape", "d_occlusion")
tracker.track(image_handle)
for _ in range(100):
    print(tracker.output())
    time.sleep(0.025)
