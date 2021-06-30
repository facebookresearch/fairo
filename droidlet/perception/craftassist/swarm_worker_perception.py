from droidlet.perception.craftassist import heuristic_perception
from droidlet.perception.craftassist.low_level_perception import LowLevelMCPerception
from droidlet.lowlevel.minecraft.mc_util import XYZ, IDM, to_block_pos, pos_to_np, euclid_dist, diag_adjacent

class SwarmLowLevelMCPerception(LowLevelMCPerception):
    def __init__(self, agent, perceive_freq=5):
        super(SwarmLowLevelMCPerception, self).__init__(agent, perceive_freq)
    
    def perceive(self, force=False):
        """
        Every n agent_steps (defined by perceive_freq), update in agent memory
        location/pose of all agents, players, mobs; item stack positions and
        changed blocks.

        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)
        """
        # FIXME (low pri) remove these in code, get from sql
        self.agent.pos = to_block_pos(pos_to_np(self.agent.get_player().pos))

        # if self.agent.count % self.perceive_freq == 0 or force:
        #     for mob in self.agent.get_mobs():
        #         if euclid_dist(self.agent.pos, pos_to_np(mob.pos)) < self.memory.perception_range:
        #             self.memory.set_mob_position(mob)
        #     item_stack_set = set()
        #     for item_stack in self.agent.get_item_stacks():
        #         item_stack_set.add(item_stack.entityId)
        #         if (
        #             euclid_dist(self.agent.pos, pos_to_np(item_stack.pos))
        #             < self.memory.perception_range
        #         ):
        #             self.memory.set_item_stack_position(item_stack)
        #     old_item_stacks = self.memory.get_all_item_stacks()
        #     if old_item_stacks:
        #         for old_item_stack in old_item_stacks:
        #             memid = old_item_stack[0]
        #             eid = old_item_stack[1]
        #             if eid not in item_stack_set:
        #                 self.memory.untag(memid, "_on_ground")
        #             else:
        #                 self.memory.tag(memid, "_on_ground")

        # # note: no "force"; these run on every perceive call.  assumed to be fast
        # self.update_self_memory()
        # self.update_other_players(self.agent.get_other_players())

        # # use safe_get_changed_blocks to deal with pointing
        # for (xyz, idm) in self.agent.safe_get_changed_blocks():
        #     self.on_block_changed(xyz, idm)

