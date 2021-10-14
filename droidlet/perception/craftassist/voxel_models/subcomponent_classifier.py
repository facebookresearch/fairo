"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
from multiprocessing import Queue, Process
from droidlet.perception.craftassist.heuristic_perception import all_nearby_objects
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData
from .semantic_segmentation.semseg_models import SemSegWrapper
from droidlet.base_util import blocks_list_to_npy, get_bounds


# TODO all "subcomponent" operations are replaced with InstSeg
class SubcomponentClassifierWrapper:
    """Perceive the world at a given frequency and update agent
    memory.

    creates InstSegNodes and tags them

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        model_path (str): path to the segmentation model
        perceive_freq (int): if not forced, how many Agent steps between perception.
            If 0, does not run unless forced
    """

    def __init__(self, agent, model_path, low_level_data, perceive_freq=0):
        self.agent = agent
        # Note remove the following
        self.memory = self.agent.memory
        self.perceive_freq = perceive_freq
        self.boring_blocks = low_level_data["boring_blocks"]
        self.passable_blocks = low_level_data["passable_blocks"]
        if model_path is not None:
            self.subcomponent_classifier = SubComponentClassifier(voxel_model_path=model_path)
            self.subcomponent_classifier.start()
        else:
            self.subcomponent_classifier = None

    def perceive(self, force=False):
        """
        Run the classifiers in the world and get the resulting labels

        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)

        """
        perceive_info = {}
        if self.perceive_freq == 0 and not force:
            return CraftAssistPerceptionData()
        if self.perceive_freq > 0 and self.agent.count % self.perceive_freq != 0 and not force:
            return CraftAssistPerceptionData()
        if self.subcomponent_classifier is None:
            return CraftAssistPerceptionData()
        # TODO don't all_nearby_objects again, search in memory instead
        to_label = []
        # add all blocks in marked areas
        for pos, radius in self.agent.areas_to_perceive:
            for obj in all_nearby_objects(self.agent.get_blocks, pos, self.boring_blocks, self.passable_blocks, radius):
                to_label.append(obj)
        # add all blocks near the agent
        for obj in all_nearby_objects(self.agent.get_blocks, self.agent.pos, self.boring_blocks, self.passable_blocks):
            to_label.append(obj)

        for obj in to_label:
            self.subcomponent_classifier.block_objs_q.put(obj)

        # everytime we try to retrieve as many recognition results as possible
        while not self.subcomponent_classifier.loc2labels_q.empty():
            loc2labels, obj = self.subcomponent_classifier.loc2labels_q.get()
            loc2ids = dict(obj)
            label2blocks = {}

            def contaminated(blocks):
                """
                Check if blocks are still consistent with the current world
                """
                mx, Mx, my, My, mz, Mz = get_bounds(blocks)
                yzxb = self.agent.get_blocks(mx, Mx, my, My, mz, Mz)
                for b, _ in blocks:
                    x, y, z = b
                    if loc2ids[b][0] != yzxb[y - my, z - mz, x - mx, 0]:
                        return True
                return False

            for loc, labels in loc2labels.items():
                b = (loc, loc2ids[loc])
                for l in labels:
                    if l in label2blocks:
                        label2blocks[l].append(b)
                    else:
                        label2blocks[l] = [b]
            perceive_info["labeled_blocks"] = {}
            for l, blocks in label2blocks.items():
                ## if the blocks are contaminated we just ignore
                if not contaminated(blocks):
                    locs = [loc for loc, idm in blocks]
                    perceive_info["labeled_blocks"][l] = locs

        return CraftAssistPerceptionData(labeled_blocks=perceive_info["labeled_blocks"])


class SubComponentClassifier(Process):
    """
    A classifier class that calls a voxel model to output object tags.
    """

    def __init__(self, voxel_model_path=None):
        super().__init__()

        if voxel_model_path is not None:
            logging.info(
                "SubComponentClassifier using voxel_model_path={}".format(voxel_model_path)
            )
            self.model = SemSegWrapper(voxel_model_path)
        else:
            raise Exception("specify a segmentation model")

        self.block_objs_q = Queue()  # store block objects to be recognized
        self.loc2labels_q = Queue()  # store loc2labels dicts to be retrieved by the agent
        self.daemon = True

    def run(self):
        """
        The main recognition loop of the classifier
        """
        while True:  # run forever
            tb = self.block_objs_q.get(block=True, timeout=None)
            loc2labels = self._watch_single_object(tb)
            self.loc2labels_q.put((loc2labels, tb))

    def _watch_single_object(self, tuple_blocks):
        """
        Input: a list of tuples, where each tuple is ((x, y, z), [bid, mid]). This list
               represents a block object.
        Output: a dict of (loc, [tag1, tag2, ..]) pairs for all non-air blocks.
        """

        def get_tags(p):
            """
            convert a list of tag indices to a list of tags
            """
            return [self.model.tags[i][0] for i in p]

        def apply_offsets(cube_loc, offsets):
            """
            Convert the cube location back to world location
            """
            return (cube_loc[0] + offsets[0], cube_loc[1] + offsets[1], cube_loc[2] + offsets[2])

        np_blocks, offsets = blocks_list_to_npy(blocks=tuple_blocks, xyz=True)

        pred = self.model.segment_object(np_blocks)

        # convert prediction results to string tags
        return dict([(apply_offsets(loc, offsets), get_tags([p])) for loc, p in pred.items()])

    def recognize(self, list_of_tuple_blocks):
        """
        Multiple calls to _watch_single_object
        """
        tags = dict()
        for tb in list_of_tuple_blocks:
            tags.update(self._watch_single_object(tb))
        return tags
