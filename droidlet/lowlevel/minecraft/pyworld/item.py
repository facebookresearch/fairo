"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.base_util import Pos
from droidlet.lowlevel.minecraft.pyworld.utils import make_pose


class GettableItem:
    def __init__(self, typeName, entityId=None, pos=None, idm=(0, 0), properties=[]):
        self.entityId = entityId
        self.typeName = typeName
        self.pos = pos or Pos()
        self.holder_entityId = -1
        # TODO: remove these, let agent get it from the typeName
        # machinery to make sure typeName and bid, meta match up in mc
        self.id = idm[0]
        self.meta = idm[1]
        # properties is a list of tuples of the form (predicate_text, object_text)
        self.properties = properties

    def update_position(self, x, y, z):
        self.pos = Pos(x, y, z)

    def add_to_world(self, world):
        self.entityId = world.new_eid(entityId=self.entityId)
        x, y, z, _, _ = make_pose(world.sl, world.sl, height_map=world.get_height_map())
        x, y, z = world.from_npy_coords((x, y, z))
        self.update_position(x, y, z)
        world.items[self.entityId] = self

    def get_info(self):
        info = {
            "entityId": self.entityId,
            "typeName": self.typeName,
            "id": self.id,
            "meta": self.meta,
            "pos": self.pos,
            "x": self.pos.x,
            "y": self.pos.y,
            "z": self.pos.z,
            "holder_entityId": self.holder_entityId,
            "properties": self.properties,
        }
        return info

    def attach_to_entity(self, entityId):
        """
        record that self is in inventory of the entity with entityId
        """
        self.holder_entityId = entityId
