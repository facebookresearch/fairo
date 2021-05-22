from droidlet.base_util import POINT_AT_TARGET
from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.interpreter import interpret_relative_direction


class PointTargetInterpreter:
    def __call__(self, interpreter, speaker, d) -> POINT_AT_TARGET:
        if d.get("location") is None:
            # TODO other facings
            raise ErrorWithResponse("I am not sure where you want me to point")
        # TODO: We might want to specifically check for BETWEEN/INSIDE, I'm not sure
        mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, d["location"])
        steps, reldir = interpret_relative_direction(interpreter.agent, d)
        loc, _ = interpreter.subinterpret["specify_locations"](
            interpreter, speaker, mems, steps, reldir
        )
        return self.point_to_region(loc)

    def point_to_region(self, loc):
        assert len(loc) == 3, "point_to_region expects a triple"
        return (loc[0], loc[1], loc[2], loc[0], loc[1], loc[2])
