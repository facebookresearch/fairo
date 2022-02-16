"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData


class DetectionWrapper:
    """Detect objects using voxel and language input and return updates.

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        model_path (str): path to the segmentation model
        text_span (str): Any text span of reference object
    """

    def __init__(self, agent, model_path):
        self.agent = agent
        if model_path is not None:
            logging.info("Using detection_model_path={}".format(model_path))
            # TODO: Yuxuan to add detection model wrapper / querier here
            self.detection_model = None  # DetectionModelWrapper(model_path, text_span)
        else:
            self.detection_model = None

    def perceive(self, text_span=None):
        """
        Run the detection classifier and get the resulting voxel predictions

        Args:
            text_span (str): text span of a reference object from logical form. This is
            used to detect a specific reference object.

        """
        if text_span is None:
            return CraftAssistPerceptionData()
        if self.detection_model is None:
            return CraftAssistPerceptionData()
        voxel_predictions = self.detection_model.parse()
        if not voxel_predictions:
            return CraftAssistPerceptionData()
        # TODO: write a method to parse through voxels and check if they pass a threshold
        updated_voxel_predictions = voxel_predictions
        # TODO: check if voxels pass a certain threshold, if not return blank
        return CraftAssistPerceptionData(detected_voxels=updated_voxel_predictions)
