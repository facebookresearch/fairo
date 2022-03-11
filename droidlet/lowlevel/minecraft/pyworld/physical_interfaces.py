import numpy as np
import logging
from droidlet.perception.craftassist.rotation import look_vec
from droidlet.shared_data_structs import TICKS_PER_SEC, Time

HEAD_HEIGHT = 2
# how many internal, non-world-interacting steps agent takes before world steps:
WORLD_STEP = 10
WORLD_STEPS_PER_DAY = 480


class FakeMCTime(Time):
    def __init__(self, world):
        self.world = world

    def get_world_hour(self):
        return (self.world.count % WORLD_STEPS_PER_DAY) / WORLD_STEPS_PER_DAY

    # converts from "seconds" to internal tick
    def round_time(self, t):
        return int(TICKS_PER_SEC * t)

    def get_time(self):
        return self.world.count * TICKS_PER_SEC

    def add_tick(self, ticks=1):
        for i in range(ticks):
            self.world.step()


class FakeCPPAction:
    NAME = "NULL"

    def __init__(self, agent):
        self.agent = agent

    def action(self, *args):
        self.agent.world_interaction_occurred = True

    def __call__(self, *args):
        if hasattr(self.agent, "recorder"):
            self.agent.recorder.record_action({"name": self.NAME, "args": list(args)})
        return self.action(*args)


class Dig(FakeCPPAction):
    NAME = "dig"

    def action(self, x, y, z):
        self.agent.world_interaction_occurred = True
        dug = self.agent.world.dig((x, y, z))
        if dug:
            self.agent._changed_blocks.append(((x, y, z), (0, 0)))
            return True
        else:
            return False


class SendChat(FakeCPPAction):
    NAME = "send_chat"

    def action(self, chat):
        self.agent.world_interaction_occurred = True
        logging.info("FakeAgent.send_chat: {}".format(chat))
        self.agent._outgoing_chats.append(chat)


class SetHeldItem(FakeCPPAction):
    NAME = "set_held_item"

    def action(self, arg):
        self.agent.world_interaction_occurred = True
        try:
            d, m = arg
            self.agent._held_item = (d, m)
        except TypeError:
            self.agent._held_item = (arg, 0)


class StepPosX(FakeCPPAction):
    NAME = "step_pos_x"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (1, 0, 0)


class StepNegX(FakeCPPAction):
    NAME = "step_neg_x"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (-1, 0, 0)


class StepPosZ(FakeCPPAction):
    NAME = "step_pos_z"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, 0, 1)


class StepNegZ(FakeCPPAction):
    NAME = "step_neg_z"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, 0, -1)


class StepPosY(FakeCPPAction):
    NAME = "step_pos_y"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, 1, 0)


class StepNegY(FakeCPPAction):
    NAME = "step_neg_y"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, -1, 0)


class StepForward(FakeCPPAction):
    NAME = "step_forward"

    def action(self):
        self.agent.world_interaction_occurred = True
        dx, dy, dz = self.agent._look_vec
        self.agent.pos += (dx, 0, dz)


class TurnAngle(FakeCPPAction):
    NAME = "turn_angle"

    def action(self, angle):
        self.agent.world_interaction_occurred = True
        if angle == 90:
            self.agent.turn_left()
        elif angle == -90:
            self.agent.turn_right()
        else:
            raise ValueError("bad angle={}".format(angle))


# FIXME!
class TurnLeft(FakeCPPAction):
    NAME = "turn_left"

    def action(self):
        self.agent.world_interaction_occurred = True
        old_l = (self.agent._look_vec[0], self.agent._look_vec[1])
        idx = self.agent.CCW_LOOK_VECS.index(old_l)
        new_l = self.agent.CCW_LOOK_VECS[(idx + 1) % len(self.agent.CCW_LOOK_VECS)]
        self.agent._look_vec[0] = new_l[0]
        self.agent._look_vec[2] = new_l[2]


# FIXME!
class TurnRight(FakeCPPAction):
    NAME = "turn_right"

    def action(self):
        self.agent.world_interaction_occurred = True
        old_l = (self.agent._look_vec[0], self.agent._look_vec[1])
        idx = self.agent.CCW_LOOK_VECS.index(old_l)
        new_l = self.agent.CCW_LOOK_VECS[(idx - 1) % len(self.agent.CCW_LOOK_VECS)]
        self.agent._look_vec[0] = new_l[0]
        self.agent._look_vec[2] = new_l[2]


class PlaceBlock(FakeCPPAction):
    NAME = "place_block"

    def action(self, x, y, z):
        self.agent.world_interaction_occurred = True
        block = ((x, y, z), self.agent._held_item)
        self.agent.world.place_block(block)
        self.agent._changed_blocks.append(block)
        return True


class LookAt(FakeCPPAction):
    NAME = "look_at"

    def action(self, x, y, z):
        self.agent.world_interaction_occurred = True
        look_vec = np.array(
            [x - self.agent.pos[0], y - self.agent.pos[1] - HEAD_HEIGHT, z - self.agent.pos[2]]
        )
        self.agent.set_look_vec(*look_vec.tolist())


class SetLook(FakeCPPAction):
    NAME = "set_look"

    def action(self, yaw, pitch):
        self.agent.world_interaction_occurred = True
        a = look_vec(yaw, pitch)
        self.agent.set_look_vec(a[0], a[1], a[2])


class Craft(FakeCPPAction):
    NAME = "craft"

    def action(self):
        raise NotImplementedError()


def init_agent_interfaces(agent):
    agent.dig = Dig(agent)
    agent.send_chat = SendChat(agent)
    agent.set_held_item = SetHeldItem(agent)
    agent.step_pos_x = StepPosX(agent)
    agent.step_neg_x = StepNegX(agent)
    agent.step_pos_z = StepPosZ(agent)
    agent.step_neg_z = StepNegZ(agent)
    agent.step_pos_y = StepPosY(agent)
    agent.step_neg_y = StepNegY(agent)
    agent.step_forward = StepForward(agent)
    agent.turn_angle = TurnAngle(agent)
    agent.turn_left = TurnLeft(agent)
    agent.turn_right = TurnRight(agent)
    agent.set_look = SetLook(agent)
    agent.look_at = LookAt(agent)
    agent.place_block = PlaceBlock(agent)
