"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Tuple, Dict, Any, Optional

from droidlet.dialog.dialogue_objects import (
    DialogueObject,
    ReferenceObjectInterpreter,
    FilterInterpreter,
    interpret_reference_object,
)

from .condition_helper import MCConditionInterpreter
from .attribute_helper import MCAttributeInterpreter


class DummyInterpreter(DialogueObject):
    def __init__(self, speaker: str, **kwargs):
        super().__init__(**kwargs)
        self.speaker = speaker
        self.provisional: Dict = {}
        self.action_dict_frozen = False
        self.loop_data = None
        self.subinterpret = {
            "attribute": MCAttributeInterpreter(),
            "filters": FilterInterpreter(),
            "reference_objects": ReferenceObjectInterpreter(interpret_reference_object),
            "condition": MCConditionInterpreter(),
        }
        self.action_handlers = {}  # noqa

    def step(self) -> Tuple[Optional[str], Any]:
        return None, None
