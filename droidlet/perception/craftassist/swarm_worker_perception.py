from droidlet.perception.craftassist.low_level_perception import LowLevelMCPerception
from droidlet.lowlevel.minecraft.mc_util import XYZ, IDM, to_block_pos, pos_to_np

class SwarmLowLevelMCPerception(LowLevelMCPerception):
    def __init__(self, agent, perceive_freq=5):
        super(SwarmLowLevelMCPerception, self).__init__(agent, perceive_freq)
    
    def perceive(self, force=False):
        """
        minimal perception modules for swarmworker
        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)
        """
        # FIXME (low pri) remove these in code, get from sql
        self.agent.pos = to_block_pos(pos_to_np(self.agent.get_player().pos))

