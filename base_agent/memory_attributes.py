"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from memory_filters import get_property_value


# attribute has function signature list(mems) --> list(value)
class Attribute:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, mems):
        raise NotImplementedError("Implemented by subclass")


class TableColumn(Attribute):
    def __init__(self, agent, attribute, get_all=False):
        super().__init__(agent)
        self.attribute = attribute
        # if this is true, the value will be a list of all outputs
        # for attributes where one mem can have multiple values
        self.get_all = get_all

    def __call__(self, mems):
        return [
            get_property_value(self.agent.memory, mem, self.attribute, get_all=self.get_all)
            for mem in mems
        ]

    def __repr__(self):
        return "Attribute: " + self.attribute


class ListAttribute(Attribute):
    def __init__(self, agent, attributes):
        super().__init__(agent)
        self.attributes = attributes

    def __call__(self, mems):
        return list(zip(*[a(mems) for a in self.attributes]))

    def __repr__(self):
        return "List Attribute: " + self.attributes.format()


class LinearExtentAttribute(Attribute):
    """ 
    computes the (perhaps signed) length between two points in space.  
    if "relative_direction"=="AWAY", unsigned length
    if "relative_direction" in ["LEFT", "RIGHT" ...] projected onto a special direction 
         and signed.  the "arrow" goes from "source" to "destination",
         e.g. if destination is more LEFT than source, "LEFT" will be positive
    if "relative_direction" in ["INSIDE", "OUTSIDE"], signed length is shifted towards zero
         so that 0 is at the boundary of the source.
         This is not implemented yet FIXME!!
    One of the two points in space is given by the positions of a reference object
    either given directly as a memory, or given as FILTERs to search
    the other is the list element input into the call
    """

    def __init__(self, agent, location_data, mem=None, fixed_role="source"):
        super().__init__(agent)
        self.coordinate_transforms = agent.coordinate_transforms
        self.location_data = location_data
        self.fixed_role = fixed_role

        self.frame = location_data.get("frame") or "AGENT"

        # TODO generalize/formalize this
        # TODO: currently stores look vecs/orientations at creation,
        #     build mechanism to update orientations, e.g. if giving directions
        #     "first you turn left, then go 7 steps forward, turn right, go 7 steps forward"
        #     need this in grammar too
        # TODO store fixed pitch/yaw etc. with arxiv memories, not raw
        try:
            if self.frame == "AGENT":
                # TODO handle this appropriately!
                yaw, pitch = agent.memory._db_read(
                    "SELECT yaw, pitch FROM ReferenceObjects WHERE uuid=?", agent.memory.self_memid
                )[0]
            elif self.frame == "ABSOLUTE":
                yaw, pitch = self.coordinate_transforms.yaw_pitch(
                    self.coordinate_transforms.DIRECTIONS["FRONT"]
                )
            # this is another player/agent; it is assumed that the frame has been replaced with
            # with the eid of the player/agent
            else:
                # TODO error if eid not found; but then parent/helper should have caught it?
                # TODO error properly if eid is a ref object, but pitch or yaw are null
                yaw, pitch = agent.memory._db_read(
                    "SELECT yaw, pitch FROM ReferenceObjects WHERE eid=?", self.frame
                )[0]
        except:
            # TODO handle this better
            raise Exception(
                "Unable to find the yaw, pitch in the given frame; maybe can't find the eid?"
            )

        self.yaw = yaw
        self.pitch = pitch
        self.mem = mem
        self.searcher = "mem"
        # put a "NULL" mem in input to not build a searcher
        if not self.mem:
            self.searcher = self.location_data.get("filter")
        #            d = self.location_data.get(fixed_role)
        #            if d:
        #                self.searcher = BasicMemorySearcher(search_data=d)
        if not self.mem and (self.searcher == "mem" or not self.searcher):
            raise Exception("Bad linear attribute data, no memory and no searcher specified")

    def extent(self, source, destination):
        # source and destination are arrays in this function
        # arrow goes from source to destination:
        diff = np.subtract(source, destination)
        if self.location_data["relative_direction"] in ["INSIDE", "OUTSIDE"]:
            raise Exception("inside and outside not yet implemented in linear extent")
        if self.location_data["relative_direction"] in [
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
            "FRONT",
            "BACK",
        ]:
            reldir_vec = self.coordinate_transforms.DIRECTIONS[
                self.location_data["relative_direction"]
            ]
            # this should be an inverse transform so we set inverted=True
            dir_vec = self.coordinate_transforms.transform(
                reldir_vec, self.yaw, self.pitch, inverted=True
            )
            return diff @ dir_vec
        else:  # AWAY
            return np.linalg.norm(diff)

    def __call__(self, mems):
        if not self.mem:
            fixed_mem, _ = self.searcher()
            fixed_mem = fixed_mem[0]
        #            fixed_mem = self.searcher.search(self.agent.memory)
        # FIXME!!! handle mem not found
        else:
            fixed_mem = self.mem
        # FIMXE TODO store and use an arxiv if we don't want position to track!
        if self.fixed_role == "source":
            return [self.extent(fixed_mem.get_pos(), mem.get_pos()) for mem in mems]
        else:
            return [self.extent(mem.get_pos(), fixed_mem.get_pos()) for mem in mems]

    def __repr__(self):
        return "Attribute: " + str(self.location_data)


class LookRayDistance(Attribute):
    """ 
    computes the (perhaps signed) distance between a ref_obj_node and a ray given by an agent's
    look and pos.  The agent's name or eid is input to contructor (in addition to the agent running 
    this attribute)

    """

    def __init__(self, agent, eid):
        super().__init__(agent)
        # TODO: currently stores look vecs/orientations at creation,
        try:
            x, y, z, yaw, pitch = agent.memory._db_read(
                "SELECT x, y, z yaw, pitch FROM ReferenceObjects WHERE eid=?", eid
            )[0]
        except:
            # TODO handle this better
            raise Exception(
                "Unable to find the yaw, pitch of viewing entity when building LookRayDistance"
            )
        self.yaw = yaw
        self.pitch = pitch
        self.pos = np.array([x, y, z])
        self.coordinate_transforms = agent.coordinate_transforms

    def __call__(self, mems):
        try:
            positions = [mem.get_pos() for mem in mems]
        except:
            raise Exception("a memory input to LookRayDistance does not .get_pos() properly")

        rotated_coords = [
            self.coordinate_transforms.transform(np.array(p) - self.pos, self.yaw, self.pitch)
            for p in positions
        ]
        LEFT = self.coordinate_transforms.DIRECTIONS["LEFT"]
        UP = self.coordinate_transforms.DIRECTIONS["UP"]
        return [((c @ LEFT) ** 2 + (c @ UP) ** 2) ** 0.5 for c in rotated_coords]

    def __repr__(self):
        return "LookRayDistance"


class ComparatorAttribute(Attribute):
    def __init__(
        self, agent, comparison_type="EQUAL", value_left=None, value_right=None, epsilon=0
    ):
        super().__init__(agent)
        self.comparison_type = comparison_type
        # at least one of these should be an Attribute; the other
        # is allowed to be a Value (but can also be an attribute)
        # FIXME error if both are values or if one doesn't exist
        self.value_left = value_left
        self.value_right = value_right
        self.epsilon = epsilon

    # raise errors if no value left or right?
    # raise errors if strings compared with > < etc.?
    # FIXME handle type mismatches
    # TODO less types, use NotCondition
    # TODO MOD_EQUAL, MOD_CLOSE
    def __call__(self, mems):
        if isinstance(self.value_left, Attribute):
            values_left = self.value_left(mems)
        else:
            value_left = self.value_left.get_value()
            values_left = [value_left] * len(mems)

        if isinstance(self.value_right, Attribute):
            values_right = self.value_right(mems)
        else:
            value_right = self.value_right.get_value()
            values_right = [value_right] * len(mems)

        # FIXME?
        if not self.value_left:
            return [False] * len(mems)
        if not value_right:
            return [False] * len(mems)

        f = COMPARATOR_FUNCTIONS.get(self.comparison_type)
        if f:
            return [f(values_left[i], values_right[i], self.epsilon) for i in range(len(mems))]
        else:
            raise Exception("unknown comparison type {}".format(self.comparison_type))


COMPARATOR_FUNCTIONS = {
    "GREATER_THAN_EQUAL": lambda l, r, e: l >= r,
    "GREATER_THAN": lambda l, r, e: l > r,
    "EQUAL": lambda l, r, e: l == r,
    "NOT_EQUAL": lambda l, r, e: l != r,
    "LESS_THAN": lambda l, r, e: l < r,
    "CLOSE_TO": lambda l, r, e: abs(l - r) <= e,
    "LESS_THAN_EQUAL": lambda l, r, e: l <= r,
}
