"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.memory.memory_filters import Attribute

# TODO sqlize these?
# FIXME!!!!! rn this will not accurately count voxels in
# InstSeg objects with given properties;
# it only gets the tag from the VoxelObject
class VoxelCounter(Attribute):
    def __init__(self, memory, block_data=[]):
        """Count voxels satisfying the properties in block_data
        block_data is a list of dicts {"pred_text":<pred>, "obj_text", <obj>}
        "pred_text" is optional in each dict
        if the mem is not a voxel, has *all* the tags is counted.
        # TODO FILTERS"""
        super().__init__(memory)
        self.block_data = block_data

    def __call__(self, mems):
        counts = []

        def allowed_idm(idm):
            return True

        if self.block_data:
            # FIXME: allow more general queries
            block_type_triples = (
                "("
                + " AND ".join(
                    ["({}={})".format(d["pred_text"], d["obj_text"]) for d in self.block_data]
                )
                + ")"
            )
            _, block_type_mems = self.memory.basic_search(
                "SELECT MEMORY FROM BlockType WHERE " + block_type_triples
            )
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
                    triple_objs = [t[2] for t in self.memory.get_triples(subj=mems[i].memid)]
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
