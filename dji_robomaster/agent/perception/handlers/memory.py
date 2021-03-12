"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
import torch
import torchvision.models as models
import numpy as np
import loco_memory
import logging
from torchvision import transforms
from dlevent import sio


class MemoryHandler(AbstractHandler):
    """Class for saving the state of the world parsed by all perceptual models
    to memory.

    The MemoryHandler also performs object reidentification and is responsible for maintaining
    a consistent state of the world using the output of all other perception handlers.

    Args:
        agent (LocoMCAgent): reference to the agent.
    """

    def __init__(self, agent):
        self.agent = agent
        self.object_id_counter = 1
        self.dedupe_model = models.resnet18(pretrained=True).cuda()
        self.layer = self.dedupe_model._modules.get("avgpool")
        self.dedupe_model.eval()
        self.init_event_handlers()

    def init_event_handlers(self):
        @sio.on("get_memory_objects")
        def objects_in_memory(sid):
            objects = loco_memory.DetectedObjectNode.get_all(self.agent.memory)
            for o in objects:
                del o["feature_repr"]
            self.agent.dashboard_memory["objects"] = objects
            sio.emit("updateState", {"memory": self.agent.dashboard_memory})

    def get_feature_repr(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        t_img = normalize(to_tensor(img)).unsqueeze(0).cuda()
        my_embedding = torch.zeros(512)

        def copy_data(m, i, o):
            my_embedding.copy_(torch.flatten(o).data)

        h = self.layer.register_forward_hook(copy_data)
        with torch.no_grad():
            self.dedupe_model(t_img)
        h.remove()
        return my_embedding

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

    def is_novel(self, rgb, world_obj, dedupe):
        """this is long-term tracking (not in-frame). it does some feature
        matching to figure out if we've seen this exact instance of object
        before.

        It uses the cosine similarity of conv features and (separately),
        distance between two objects
        """
        # Treat all objects as novel if dedupe is turned off.
        if not dedupe:
            return True
        world_obj.feature_repr = self.get_feature_repr(world_obj.get_masked_img())
        cos = torch.nn.CosineSimilarity(dim=1)
        is_novel = True
        objects = loco_memory.DetectedObjectNode.get_all(self.agent.memory)
        # Future = use all object features to calculate novelty
        for x in objects:
            score = cos(x["feature_repr"].unsqueeze(0), world_obj.feature_repr.unsqueeze(0))
            dist = np.linalg.norm(np.asarray(world_obj.xyz[:2]) - np.asarray(x["xyz"][:2]))
            logging.debug(
                "Similarity {}.{} = {}, {}".format(world_obj.label, x["label"], score.item(), dist)
            )
            # FIXME pick best match?
            if self.is_match(score.item(), dist):
                is_novel = False
                world_obj.eid = x["eid"]
            # delete so that we can emit it (i.e. x becomes serializable as-is)
            del x["feature_repr"]
        # todo: only emit new or updated objects, rather than emitting everything everything
        self.agent.dashboard_memory["objects"] = objects
        sio.emit("updateState", {"memory": self.agent.dashboard_memory})
        logging.info("world object {}, is_novel {}".format(world_obj.label, is_novel))
        return is_novel

    def handle(self, rgb, objects, dedupe=True):
        """run the memory handler for the current rgb, objects detected.

        This is also where each WorldObject is assigned a unique entity id (eid).

        Args:
            rgb (np.array): current RGB image for which the handlers are being run
            objects (list[WorldObject]): a list of all WorldObjects detected
            dedupe (boolean): boolean to indicate whether to deduplicate objects or not. If False, all objects
            are considered to be new/novel objects that the agent hasn't seen before. If True, the handler tries to
            identify whether it has seen a WorldObject before and updates it if it has.
        """
        logging.info("In MemoryHandler ... ")
        for i, x in enumerate(objects):
            if self.is_novel(rgb, x, dedupe):
                x.eid = self.object_id_counter
                self.object_id_counter = self.object_id_counter + 1
                x.save_to_memory(self.agent)
                logging.info(
                    f"Instance#{i} ({x.label}) is at location: ({np.around(np.array(x.xyz), 2)}), Center:({x.center})"
                )
            else:
                x.save_to_memory(self.agent, update=True)
        logging.debug(
            f"Current Locobot pos (x_standard, z_standard, yaw) is: {self.agent.mover.get_base_pos_in_canonical_coords()}"
        )
