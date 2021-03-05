"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import csv
from typing import Tuple, Optional

from dialogue_stack import DialogueStack
from .dialogue_objects import DialogueObject, Say
from abc import ABC, abstractmethod


class DialogueManager(ABC):
    """
    | The current control flow of dialogue is:
    | 1.  A chat comes in and Dialogue manager reads it or the bot triggers a
    |    dialogue because of memory/perception/task state
    | 2.  The dialogue manager puts a DialogueObject on the DialogueStack.
    | 3.  The DialogueStack calls .step() which in turn calls the DialogueObject.step()
    |    that performs some action as implemented in the step method. The step could
    |    also possibly interact with the agent's memory. And finally the step() makes
    |    a call and decides if the DialogueObject is finished.
    |
    |
    | -   The step() returns a string:  maybe_chat, a dict: maybe_data.
    | -   The step()'s outputs are read by the manager which can decide to put another
    |    DialogueObject on the stack.

    | The maybe_data from the output of the dialogue object's step() can
    | contain a 'push' key; this overrides the manager's decision on what to push to
    | the stack.

    | Control flow for interpreter and clarification:
    | The interpreter is also a kind of DialogueObject, and a clarification step is
    | the interpreter returning control to the DialogueManager, which pushes a
    | ConfirmTask or ConfirmReferenceObject as a DialogueObject onto the DialogueStack.

    args:
        agent: a droidlet agent
        model: a (perhaps ML) model used by and the model used for manager.
    """
    def __init__(self, agent, model):
        self.agent = agent
        self.dialogue_stack = DialogueStack(agent, agent.memory)
        self.model = model

    def get_safety_words(self, safety_words_path):
        """Read a list of safety words to prevent abuse.
        """
        with open(safety_words_path) as f:
            safety_lines = f.readlines()
        safety_words = []
        for l in safety_lines:
            w = l.strip("\n").lower()
            if w != "" and w[0] != "<" and w[0] != "#":
                safety_words.append(w)
        return safety_words

    def is_safe(self, chat):
        """Check that chat does not contain unsafe words.
        """
        safety_set = set(self.safety_words)
        cmd_set = set(chat.lower().split())
        notsafe = len(cmd_set & safety_set) > 0
        return not notsafe

    def step(self, chat: Tuple[str, str]):
        """Process a chat and step through the dialogue manager task stack.

        The chat is given as input to the model, which returns a logical form.
        The logical form is converted to a dialogue object, which allows the interpreter to
        handle the action.
        Unless empty, the dialogue object is put on the dialogue stack.
        Then the DialogueStack calls .step(), which in turn calls the DialogueObject.step().
        DialogueObject.step() determines whether the dialogue object has finished.

        Args:
            chat (Tuple[str, str]): Tuple of (speaker, chat). Speaker is name of speaker.
                Chat is an incoming chat command that the agent has received.
                Example: ("player_1", "build a red cube")

        """
        # check safety
        if not self.is_safe(chat[1]):
            self.dialogue_stack.append_new(Say, "Please don't be rude.")
            return

        if chat[1]:
            logging.debug("Dialogue stack pre-run_model: {}".format(self.dialogue_stack.stack))

            # NOTE: the model is responsible for not putting a new
            # object on the stack if it sees that whatever is on
            # the stack should continue.
            # TODO: Maybe we need a HoldOn dialogue object?
            obj = self.maybe_get_dialogue_obj(chat)
            if obj is not None:
                self.dialogue_stack.append(obj)

        # Always call dialogue_stack.step(), even if chat is empty
        if len(self.dialogue_stack) > 0:
            self.dialogue_stack.step()

    def maybe_get_dialogue_obj(self, chat: Tuple[str, str]) -> Optional[DialogueObject]:
        raise NotImplementedError("Must implement maybe_get_dialogue_object in subclass")
