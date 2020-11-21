"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from base_agent.dialogue_objects import AttributeInterpreter
from mc_attributes import VoxelGeometry, VoxelCounter

# this will become unnecessary with distance between
class MCAttributeInterpreter(AttributeInterpreter):
    def __call__(self, interpreter, speaker, d_attribute, get_all=False):
        if type(d_attribute) is str:
            if d_attribute.lower() == "height" or d_attribute.lower() == "width":
                return VoxelGeometry(interpreter.agent, d_attribute.lower())
            else:
                return super().__call__(interpreter, speaker, d_attribute, get_all=get_all)
        elif type(d_attribute) is dict:
            bd = d_attribute.get("num_blocks")
            if bd:
                block_data = []
                for k, v in bd.get("block_filters", {}).items():
                    block_data.append({"obj_text": v})
                return VoxelCounter(block_data)
            else:
                return super().__call__(interpreter, speaker, d_attribute, get_all=get_all)
        else:
            return super().__call__(interpreter, speaker, d_attribute, get_all=get_all)
