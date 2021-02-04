from base_agent.base_util import ErrorWithResponse
from base_agent.dialogue_objects import interpret_relative_direction

class PointTargetInterpreter:
    def __call__(self, interpreter, speaker, d):
        if d.get("location") is None:
            # TODO other facings
            raise ErrorWithResponse("I am not sure where you want me to point")
        # TODO: We might want to specifically check for BETWEEN/INSIDE, I'm not sure
        # what the +1s are in the return value
        mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, d["location"])
        return self.point_to_region(mems[0].get_point_at_target())

    def point_to_region(self, pointed_target):
        # mc pointed target is always a region represented as [xmin, ymin, zmin, xmax, ymax, zmax]
        assert len(pointed_target) == 6, "pointed target should be a region in mc"
        return pointed_target
