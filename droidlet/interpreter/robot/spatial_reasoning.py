"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np

DEFAULT_NUM_STEPS = 1


class ComputeLocations:
    def __call__(
        self,
        interpreter,
        speaker,
        mems,
        steps,
        reldir,
        repeat_num=1,
        repeat_dir=None,
        objects=[],
        padding=(1, 1, 1),
    ):
        repeat_num = max(repeat_num, len(objects))
        origin = compute_location_heuristic(mems, steps, reldir, interpreter.memory)
        if repeat_num > 1:
            raise NotImplementedError
        else:
            offsets = [(0, 0, 0)]
        #        offsets = [post_process_loc(o, interpreter) for o in offsets]
        return origin, offsets


# FIXME this can be merged with the MC version without too much work,
# main difference is speaker vs agent frame (but that is in agent.default_frame)
# There will be at least one mem in mems
def compute_location_heuristic(mems, steps, reldir, memory):
    loc = mems[0].get_pos()
    self_mem = memory.get_mem_by_id(memory.self_memid)
    if reldir is not None:
        steps = steps or DEFAULT_NUM_STEPS
        if reldir == "BETWEEN":
            loc = tuple((np.add(mems[0].get_pos(), mems[1].get_pos())) / 2)
        elif reldir == "INSIDE":
            raise NotImplementedError
        elif reldir == "NEAR":
            pass
        elif reldir == "AROUND":
            pass
        else:  # LEFT, RIGHT, etc...
            reldir_vec = memory.coordinate_transforms.DIRECTIONS[reldir]
            yaw, _ = self_mem.get_yaw_pitch()  # work in agent frame
            # we are converting a agent-frame reldir to absolute frame so we set inverted=True
            dir_vec = memory.coordinate_transforms.transform(reldir_vec, yaw, 0, inverted=True)
            loc = steps * np.array(dir_vec) + np.array(loc)
    elif steps is not None:
        loc = np.add(loc, [0, 0, steps])
    return tuple(loc)
