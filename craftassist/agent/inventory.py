"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Tuple, List
from .mc_util import IDM

ITEM_STACK_NODE = Tuple[str, int]  # (memid, count)
ITEM_QUEUE = Tuple[
    int, List[ITEM_STACK_NODE]
]  # (queue_size, [item_stack_node_1, item_stack_node_2, ...])


class Inventory:
    """Handlers for adding and removing items from the agent's 
    inventory."""

    def __init__(self):
        self.items_map = {}

    def add_item_stack(self, idm: IDM, item_stack_node: ITEM_STACK_NODE):
        if idm not in self.items_map:
            self.items_map[idm] = [0, []]
        self.items_map[idm][0] += item_stack_node[1]
        self.items_map[idm][1].append(item_stack_node)

    def remove_item_stack(self, idm: IDM, memid):
        if idm not in self.items_map or memid not in self.items_map[idm][1]:
            return
        item_stack_node = next(i for i in self.items_map[idm][1] if i[0] == memid)
        if item_stack_node:
            self.items_map[idm][0] -= item_stack_node[1]
            self.items_map[idm][1].remove(item_stack_node)

    def get_idm_from_memid(self, memid):
        for idm, item_q in self.items_map.items():
            memids = [item_stack[0] for item_stack in item_q[1]]
            if memid in memids:
                return idm
        return (0, 0)

    def get_item_stack_count_from_memid(self, memid):
        for idm, item_q in self.items_map.items():
            for item_stack in item_q[1]:
                if memid == item_stack[0]:
                    return item_stack[1]
        return 0

    def sync_inventory(self, inventory):
        pass
