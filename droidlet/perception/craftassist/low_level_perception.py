"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from typing import Tuple, List
from droidlet.base_util import to_block_pos, XYZ, IDM, pos_to_np, euclid_dist
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData


def capped_line_of_sight(agent, player_struct, cap=20):
    """Return the block directly in the entity's line of sight, or a point in the distance"""
    xsect = agent.get_player_line_of_sight(player_struct)
    if xsect is not None and euclid_dist(pos_to_np(xsect), pos_to_np(player_struct.pos)) <= cap:
        return pos_to_np(xsect)

    # default to cap blocks in front of entity
    vec = agent.coordinate_transforms.look_vec(player_struct.look.yaw, player_struct.look.pitch)
    return cap * np.array(vec) + to_block_pos(pos_to_np(player_struct.pos))


class LowLevelMCPerception:
    """Perceive the world at a given frequency and send updates back to the agent

    updates positions of other players, mobs, self, and changed blocks,
    takes this information directly from the craftassist_cuberite_utils server

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        perceive_freq (int): if not forced, how many Agent steps between perception
    """

    def __init__(self, agent, perceive_freq=5):
        self.agent = agent
        self.memory = agent.memory # NOTE: remove this once done
        self.pending_agent_placed_blocks = set()
        self.perceive_freq = perceive_freq

    def perceive(self, force=False):
        """
        Every n agent_steps (defined by perceive_freq), update in agent memory
        location/pose of all agents, players, mobs; item stack positions and
        changed blocks.

        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)
        """
        perceive_info = {}
        # FIXME (low pri) remove these in code, get from sql
        self.agent.pos = to_block_pos(pos_to_np(self.agent.get_player().pos))
        boring_blocks = self.agent.low_level_data["boring_blocks"]

        if self.agent.count % self.perceive_freq == 0 or force:
            # Find mobs in perception range
            mobs = []
            for mob in self.agent.get_mobs():
                if euclid_dist(self.agent.pos, pos_to_np(mob.pos)) < self.agent.memory.perception_range:
                    mobs.append(mob)
            perceive_info["mobs"] = mobs if mobs else None

            # Find items that can be picked by the agent, and in perception range
            all_items = set()
            in_perception_items = []
            for item_stack in self.agent.get_item_stacks():
                all_items.add(item_stack.entityId)
                if (
                    euclid_dist(self.agent.pos, pos_to_np(item_stack.pos))
                    < self.agent.memory.perception_range
                ):
                    in_perception_items.append(item_stack)
            perceive_info['agent_pickable_items'] = perceive_info.get('agent_pickable_items', {})
            perceive_info['agent_pickable_items']['in_perception_items'] = in_perception_items if in_perception_items else None
            perceive_info['agent_pickable_items']['all_items'] = all_items if all_items else None

        # note: no "force"; these run on every perceive call.  assumed to be fast
        perceive_info["agent_attributes"] = self.get_agent_player() # Get Agent attributes
        # List of other players in-game
        perceive_info["other_player_list"] = self.update_other_players(self.agent.get_other_players())
        # Changed blocks and their attributes
        perceive_info["changed_block_attributes"] = {}
        for (xyz, idm) in self.agent.safe_get_changed_blocks():
            interesting, player_placed, agent_placed = self.on_block_changed(xyz, idm, boring_blocks)
            perceive_info["changed_block_attributes"][(xyz, idm)] = [interesting, player_placed, agent_placed]

        return CraftAssistPerceptionData(mobs=perceive_info["mobs"],
                                         agent_pickable_items=perceive_info["agent_pickable_items"],
                                         agent_attributes=perceive_info["agent_attributes"],
                                         other_player_list=perceive_info["other_player_list"],
                                         changed_block_attributes=perceive_info["changed_block_attributes"])


    def get_agent_player(self):
        """Return agent's current position and attributes"""
        return self.agent.get_player()


    def update_other_players(self, player_list: List, force=False):
        """Update the location of other in-game players wrt to agent's line of sight
        Args:
            a list of [player_struct, updated location] from agent
        """
        updated_player_list = []
        for player in player_list:
            location = capped_line_of_sight(self.agent, player)
            location[1] += 1
            updated_player_list.append([player, location])
        return updated_player_list


    def on_block_changed(self, xyz: XYZ, idm: IDM, boring_blocks: Tuple[int]):
        """Update the state of the world when a block is changed."""
        # TODO don't need to do this for far away blocks if this is slowing down bot
        interesting, player_placed, agent_placed = self.mark_blocks_with_env_change(xyz, idm, boring_blocks)
        return interesting, player_placed, agent_placed


    def clear_air_surrounded_negatives(self):
        pass


    # eventually some conditions for not committing air/negative blocks
    def mark_blocks_with_env_change(self, xyz: XYZ, idm: IDM, boring_blocks: Tuple[int], agent_placed=False):
        """
        Mark the interesting blocks when any change happens in the environment
        """
        if not agent_placed:
            interesting, player_placed, agent_placed = self.is_placed_block_interesting(
                xyz, idm[0], boring_blocks
            )
        else:
            interesting = True
            player_placed = False

        if not interesting:
            return None, None, None

        if agent_placed:
            try:
                self.pending_agent_placed_blocks.remove(xyz)
            except:
                pass
        return interesting, player_placed, agent_placed


    # FIXME move removal of block to parent
    def is_placed_block_interesting(self, xyz: XYZ, bid: int, boring_blocks: Tuple[int]) -> Tuple[bool, bool, bool]:
        """Return three values:
        - bool: is the placed block interesting?
        - bool: is it interesting because it was placed by a player?
        - bool: is it interesting because it was placed by the agent?
        """
        interesting = False
        player_placed = False
        agent_placed = False
        # TODO record *which* player placed it
        if xyz in self.pending_agent_placed_blocks:
            interesting = True
            agent_placed = True
        for player_struct in self.agent.get_other_players():
            if (
                euclid_dist(pos_to_np(player_struct.pos), xyz) < 5
                and player_struct.mainHand.id == bid
            ):
                interesting = True
                if not agent_placed:
                    player_placed = True
        if bid not in boring_blocks:
            interesting = True
        return interesting, player_placed, agent_placed