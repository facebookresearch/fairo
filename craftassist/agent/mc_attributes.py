"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from base_agent.memory_attributes import Attribute


class VoxelGeometry(Attribute):
    def __init__(self, agent, attribute="height"):
        """height, max_width"""
        super().__init__(agent)
        self.attribute = attribute

    def __call__(self, mems):
        bounds = [m.get_bounds() for m in mems]
        if self.attribute == "width":
            return [max(b[1] - b[0], b[5] - b[4]) for b in bounds]
        else:
            return [b[3] - b[2] for b in bounds]

    def __repr__(self):
        return "VoxelGeometry " + str(self.attribute)


# TODO sqlize these?
# FIXME!!!!! rn this will not accurately count voxels in
# InstSeg objects with given properties;
# it only gets the tag from the VoxelObject
class VoxelCounter(Attribute):
    def __init__(self, agent, block_data=[]):
        """Count voxels satisfying the properties in block_data
        block_data is a list of dicts {"pred_text":<pred>, "obj_text", <obj>}
        "pred_text" is optional in each dict
        if the mem is not a voxel, has *all* the tags is counted.
        # TODO FILTERS"""
        super().__init__(agent)
        self.block_data = block_data

    def __call__(self, mems):
        counts = []

        def allowed_idm(idm):
            return True

        if self.block_data:
            block_type_search_data = {"base_table": "BlockTypes", "triples": self.block_data}
            block_type_mems = self.agent.memory.basic_search(block_type_search_data)
            allowed_idm_list = [(b.b, b.m) for b in block_type_mems]

            def allowed_idm(idm):  # noqa
                return idm in allowed_idm_list

        for i in range(len(mems)):
            if mems[i].NODE_TYPE == "BlockObject":
                if self.block_data:
                    count = len([idm for idm in mems[i].blocks.values() if allowed_idm(idm)])
                else:
                    count = len(mems[i].blocks)
            elif mems[i].NODE_TYPE == "InstSeg":
                if self.block_data:
                    # FIXME?:
                    triple_objs = [t[2] for t in self.agent.memory.get_triples(subj=mems[i].memid)]
                    desired_objs = [t["obj_text"] for t in self.block_data]
                    if all([t in triple_objs for t in desired_objs]):
                        count = len(mems[i].blocks)
                else:
                    count = len(mems[i].blocks)
            else:
                count = 0
            counts.append(count)
        return counts

    def __repr__(self):
        return "VoxelCounter " + str(self.block_data)
