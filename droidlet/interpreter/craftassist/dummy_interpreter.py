"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Tuple, Dict, Any, Optional

from ..interpreter import (
    ReferenceObjectInterpreter,
    FilterInterpreter,
    interpret_reference_object,
    InterpreterBase,
)
from ..interpret_conditions import ConditionInterpreter
from .interpret_attributes import MCAttributeInterpreter


class DummyInterpreter:
    def __init__(self, speaker, logical_form_memid, agent_memory, memid=None, low_level_data=None):
        self.memory = agent_memory
        self.subinterpret = {
            "attribute": MCAttributeInterpreter(),
            "filters": FilterInterpreter(),
            "reference_objects": ReferenceObjectInterpreter(interpret_reference_object),
            "condition": ConditionInterpreter(),
        }
        self.action_handlers = {}  # noqa

    def step(self):
        pass
