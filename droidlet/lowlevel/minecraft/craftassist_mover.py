"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from typing import cast
from droidlet.base_util import XYZ, Pos, Look
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Item, ItemStack, Mob


def flip_x(struct, floor=False):
    x, y, z = struct.x, struct.y, struct.z
    if floor:
        x = float(np.floor(x))
        y = float(np.floor(y))
        z = float(np.floor(z))
    return Pos(-x, y, z)


def flip_look(struct):
    yaw = -np.deg2rad(struct.yaw)
    pitch = -np.deg2rad(struct.pitch)
    return Look(yaw, pitch)


def maybe_flip_x_or_look(struct, floor=False):
    """
    struct is either a Mob, a Player, a Pos, or a Look
    we make a copy with the x negated if the struct is or has a Pos
    and with the yaw and pitch negated if the struct is or has a Look
    if floor=True, will also floor pos in cagent world.
    """
    if getattr(struct, "x", None):
        return flip_x(struct)
    elif getattr(struct, "yaw", None):
        return flip_look(struct)
    elif getattr(struct, "mobType", None):
        # we keep the cagent struct as an attribute in case we want to interface with cagent again
        return Mob(
            struct.entityId, struct.mobType, flip_x(struct.pos), flip_look(struct.look), struct
        )
    elif getattr(struct, "mainHand", None):
        # we keep the cagent struct as an attribute in case we want to interface with cagent again
        return Player(
            struct.entityId,
            struct.name,
            flip_x(struct.pos, floor=floor),
            flip_look(struct.look),
            struct.mainHand,
            struct,
        )
    elif getattr(struct, "item", None):
        return ItemStack(struct.item, flip_x(struct.pos), struct.entityId)
    else:
        # maybe raise an error? or special case for None outputs?
        return struct


def struct_transform(func):
    """
    modify the input function func to return a flipped struct
    """

    def g(*args, **kwargs):
        out = func(*args, **kwargs)
        if type(out) is list:
            return [maybe_flip_x_or_look(o) for o in out]
        else:
            return maybe_flip_x_or_look(out)

    return g


class CraftassistMover:
    def __init__(self, cagent):
        self.cagent = cagent
        nongeom_cagent_fns = [
            "drop_item_stack_in_hand",
            "drop_item_in_hand",
            "drop_inventory_item_stack",
            "set_inventory_slot",
            "get_player_inventory",
            "get_incoming_chats",
            "get_inventory_item_count",
            "get_inventory_items_counts",
            "send_chat",
            "set_held_item",
            "step_forward",
            "use_entity",
            "use_item",
            "use_item_on_block",
            "craft",
            "get_world_age",
            "get_time_of_day",
            "get_vision",
            "disconnect",
        ]
        self.nongeom_cagent_fns = nongeom_cagent_fns
        for fn_name in nongeom_cagent_fns:
            setattr(self, fn_name, getattr(self.cagent, fn_name))

        # these aren't used...
        self.turn_left = self.cagent.turn_left
        self.turn_right = self.cagent.turn_right

        self.get_line_of_sight = struct_transform(self.cagent.get_line_of_sight)
        self.get_item_stacks = struct_transform(self.cagent.get_item_stacks)
        self.get_item_stack = struct_transform(self.cagent.get_item_stack)
        self.get_mobs = struct_transform(self.cagent.get_mobs)
        self.get_other_players = struct_transform(self.cagent.get_other_players)
        self.get_other_player_by_name = struct_transform(self.cagent.get_other_player_by_name)

    def get_player(self):
        return maybe_flip_x_or_look(self.cagent.get_player(), floor=True)

    @struct_transform
    def get_player_line_of_sight(self, player_struct):
        # this is a little tricky: the player_struct in droidlet space has pos x-flipped
        # and look yaw and pitch flipped.  we need the cagent's player struct to do the computation in
        # cuberite.  If the player_struct is a python object from droidlet, we assume it has a
        # cuberite cagent player struct as a member.
        if hasattr(player_struct, "cagent_struct"):
            return self.cagent.get_player_line_of_sight(player_struct.cagent_struct)
        else:
            # TODO expose cagent's player struct def and assert that data type is correct
            return self.cagent.get_player_line_of_sight(player_struct)

    def dig(self, x, y, z):
        return self.cagent.dig(-x, y, z)

    def place_block(self, x, y, z):
        return self.cagent.place_block(-x, y, z)

    def get_changed_blocks(self):
        blocks = self.cagent.get_changed_blocks()
        transformed_blocks = []
        for xyz, idm in blocks:
            transformed_blocks.append(((-xyz[0], xyz[1], xyz[2]), idm))
        return transformed_blocks

    # FIXME!! turn_angle is broken in the cagent; should be swapping here,
    # but cagent actually has it backwards
    def turn_angle(self, yaw):
        self.cagent.turn_angle(yaw)

    def set_look(self, yaw, pitch):
        self.cagent.set_look(-yaw, -pitch)

    def look_at(self, x, y, z):
        self.cagent.look_at(-x, y, z)

    def get_blocks(self, x, X, y, Y, z, Z):
        """
        returns an (Y-y+1) x (Z-z+1) x (X-x+1) x 2 numpy array B of the blocks
        in the rectanguloid with bounded by the input coordinates (including endpoints).
        Input coordinates are in droidlet coordinates; and the output array is
        in yzxb permutation, where B[0,0,0,:] corresponds to the id and meta of
        the block at x, y, z

        TODO: we don't need yzx orientation anymore...
        """
        # negate the x coordinate to shift to cuberite coords
        B = self.cagent.get_blocks(-X, -x, y, Y, z, Z)
        return np.flip(B, 2)

    # reversed to match droidlet coords
    def step_pos_x(self):
        self.cagent.step_neg_x()

    # reversed to match droidlet coords
    def step_neg_x(self):
        self.cagent.step_pos_x()

    def step_pos_y(self):
        self.cagent.step_pos_y()

    def step_neg_y(self):
        self.cagent.step_neg_y()

    def step_pos_z(self):
        self.cagent.step_pos_z()

    def step_neg_z(self):
        self.cagent.step_neg_z()

    def step_forward(self):
        self.cagent.step_forward()


############################################################################
# in minecraft, we have
#    "AWAY":  [ 0, 0, 1],
#    "FRONT": [ 0, 0, 1],
#    "BACK":  [ 0, 0,-1],
#    "LEFT":  [ 1, 0, 0],
#    "RIGHT": [-1, 0, 0],
#    "DOWN":  [ 0,-1, 0],
#    "UP":    [ 0, 1, 0],
#
# coordinates axes:
#
#         ^ y
#         |  ^ z+
#         | /
# x+ <----/
#
# and yaw and pitch are clockwise coordinate system
#
##############################################################################
#
# the methods in this file convert to and from this coordinate system to the
# droidlet standard:
# coords are (x, y, z)
# 0 yaw is x axis
#                 z+
#                 |
#                 |
#        +yaw     |   -yaw
#                 |
#    x-___________|___________x+
#                 |
#                 |
#                 z-
#
#         ^ y+
#         |     z+
#         |   /
#         | /
#         0 -----> x+
#
#             y+
#             |
#             |   +pitch
#             |
#    z-_______|________z+
#             |
#             |
#             |   -pitch
#             |
#             y-
#


def from_minecraft_xyz_to_droidlet(xyz):
    return cast(XYZ, (-xyz[0], xyz[1], xyz[2]))


def from_droidlet_xyz_to_minecraft(xyz):
    return cast(XYZ, (-xyz[0], xyz[1], xyz[2]))


def from_minecraft_look_to_droidlet(look):
    return Look(-look.yaw, -look.pitch)


def from_droidlet_look_to_craftassist(look):
    return Look(-look.yaw, -look.pitch)
