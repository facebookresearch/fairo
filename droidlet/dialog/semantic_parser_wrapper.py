"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Tuple, Optional

from droidlet.dialog.dialogue_objects import DialogueObject


class SemanticParserWrapper(object):
    def __init__(self, dialogue_object_classes, opts, dialogue_manager):
        self.dialogue_manager = dialogue_manager
        self.dialogue_objects = dialogue_object_classes

    def get_dialogue_object(self) -> Optional[DialogueObject]:
        raise NotImplementedError("Must implement get_dialogue_object in subclass")
