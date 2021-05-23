"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.interpreter import ConditionInterpreter
from .mc_stop_condition import AgentAdjacentStopCondition
from .block_helpers import get_block_type

# this will become unnecessary with distance between
class MCConditionInterpreter(ConditionInterpreter):
    def __init__(self, block_data={}):
        super().__init__()
        self.block_data = block_data
        self.condition_types["ADJACENT_TO_BLOCK_TYPE"] = self.interpret_adjacent_block

    def interpret_adjacent_block(self, interpreter, speaker, d):
        block_type = d["block_type"]
        bid, meta = get_block_type(block_type, block_data_info=self.block_data)
        return AgentAdjacentStopCondition(interpreter.memory, bid)
