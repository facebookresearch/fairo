import sys
import warnings
from droidlet.shared_data_structs import RGBDepth

from droidlet.perception.robot.perception_util import (
    random_colors,
    draw_xyz,
    get_random_color,
    get_coords,
    get_color_tag,
)

__all__ = [
    RGBDepth,
    random_colors,
    draw_xyz,
    get_random_color,
    get_coords,
    get_color_tag,
]


try:
    from droidlet.perception.robot.handlers import (
        ObjectDetection,
        FaceRecognition,
        ObjectDeduplicator,
        Detection,
        WorldObject,
        Human,
        HumanKeypointsOrdering,
        HumanPose,
        LabelPropagate,
    )
    from droidlet.perception.robot.perception import Perception

    __all__ += [
        Perception,
        Detection,
        WorldObject,
        Human,
        HumanKeypointsOrdering,
        ObjectDetection,
        HumanPose,
        FaceRecognition,
        ObjectDeduplicator,
        LabelPropagate,
    ]
except ModuleNotFoundError:
    if sys.platform == "darwin":
        warnings.warn(
            "Could not import some perception modules, likely because on OSX there's no GPU. Skipping imports"
        )
    else:
        raise
