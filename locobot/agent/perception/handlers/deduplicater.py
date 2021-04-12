"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
from locobot.agent.objects import AttributeDict
import torch
import torchvision.models as models
import numpy as np
import logging
from torchvision import transforms

class ObjectDeduplicationHandler(AbstractHandler):
    """Class for deduplicating a given set of objects from a given set of existing objects

    """

    def __init__(self):
        self.object_id_counter = 1
        self.dedupe_model = models.resnet18(pretrained=True).cuda()
        self.layer = self.dedupe_model._modules.get("avgpool")
        self.dedupe_model.eval()
        self.transforms = [
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ]
        self._temp_buffer = torch.zeros(512).pin_memory()
        def copy_data(m, i, o):
            self._temp_buffer.copy_(torch.flatten(o).data)
        self.layer.register_forward_hook(copy_data)

    def get_feature_repr(self, img):
        normalize, to_tensor = self.transforms
        t_img = normalize(to_tensor(img)).unsqueeze(0).cuda()

        with torch.no_grad():
            self.dedupe_model(t_img)
        return self._temp_buffer.clone()

    # Not accounting for moving objects
    def is_match(self, score, dist):
        score_thresh = 0.95
        dist_thresh = 0.6
        if score > score_thresh and dist > dist_thresh:
            return False  # Similar object, different places.
        elif score > score_thresh and dist < dist_thresh:
            return True  # same object, same place
        else:
            return False

    def is_novel(self, current_object, previous_objects):
        """this is long-term tracking (not in-frame). it does some feature
        matching to figure out if we've seen this exact instance of object
        before.

        It uses the cosine similarity of conv features and (separately),
        distance between two previous_objects

        Args:
            current_object (WorldObject): current object to compare
            previous_objects (List[WorldObject]): all previous objects to compare to
            

        """
        current_object.feature_repr = self.get_feature_repr(current_object.get_masked_img())
        cos = torch.nn.CosineSimilarity(dim=1)
        is_novel = True

        # Future = use all object features to calculate novelty
        for previous_object in previous_objects:
            if isinstance(previous_object, dict):
                previous_object = AttributeDict(previous_object)
            if previous_object.feature_repr is not None and current_object.feature_repr is not None:
                score = cos(previous_object.feature_repr.unsqueeze(0), current_object.feature_repr.unsqueeze(0))
                dist = np.linalg.norm(np.asarray(current_object.xyz[:2]) - np.asarray(previous_object.xyz[:2]))
                logging.debug(
                    "Similarity {}.{} = {}, {}".format(current_object.label, previous_object.label, score.item(), dist)
                )
                # FIXME pick best match?
                if self.is_match(score.item(), dist):
                    is_novel = False
                    current_object.eid = previous_object.eid
        logging.info("world object {}, is_novel {}".format(current_object.label, is_novel))
        return is_novel

    def handle(self, current_objects, previous_objects):
        """run the deduplication for the current objects detected.

        This is also where each WorldObject is assigned a unique entity id (eid).

        Args:
            current_objects (list[WorldObject]): a list of all WorldObjects detected in the current frame
            previous_objects (list[WorldObject]): a list of all previous WorldObjects ever detected
        """
        logging.info("In ObjectDeduplicationHandler ... ")
        new_objects = []
        updated_objects = []
        for current_object in current_objects:
            if self.is_novel(current_object, previous_objects):
                current_object.eid = self.object_id_counter
                self.object_id_counter = self.object_id_counter + 1
                new_objects.append(current_object)

                logging.info(
                    f"Instance ({current_object.label}) is at location: "
                    f"({np.around(np.array(current_object.xyz), 2)}),"
                    f" Center:({current_object.center})"
                )
            else:
                updated_objects.append(current_object)

        return new_objects, updated_objects
