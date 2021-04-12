"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from typing import Tuple, Optional
import preprocess
from dialogue_stack import DialogueStack
from .dialogue_objects import DialogueObject, Say


class DialogueManager(object):
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
        """Read a set of safety words to prevent abuse."""
        with open(safety_words_path) as f:
            safety_lines = f.readlines()
        safety_words = set()
        for l in safety_lines:
            w = l.strip("\n").lower()
            if w != "" and w[0] != "<" and w[0] != "#":
                safety_words.add(w)
        return safety_words

    def is_safe(self, chat):
        """Check that chat does not contain any word from the
        safety check list.
        """
        cmd_set = set(chat.lower().split())
        notsafe = len(cmd_set & self.safety_words) > 0
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
        # chat is a single line command
        speaker, chatstr = chat
        # tokenize the chat and get list of sentences to parse.
        preprocessed_chatstrs = preprocess.preprocess_chat(chatstr)

        # check safety for each chat first
        for preprocessed_chatstr in preprocessed_chatstrs:
            if not self.is_safe(preprocessed_chatstr):
                self.dialogue_stack.append_new(Say, "Please don't be rude.")
                return

        if preprocessed_chatstrs:
            logging.debug("Dialogue stack pre-run_model: {}".format(self.dialogue_stack.stack))

            # NOTE: the model is responsible for not putting a new
            # object on the stack if it sees that whatever is on
            # the stack should continue.
            # TODO: Maybe we need a HoldOn dialogue object?
            obj = self.maybe_get_dialogue_obj(speaker=speaker, chat_list=preprocessed_chatstrs)
            if obj is not None:
                self.dialogue_stack.append(obj)

        # Always call dialogue_stack.step(), even if chat is empty
        if len(self.dialogue_stack) > 0:
            self.dialogue_stack.step()

    def maybe_get_dialogue_obj(self, chat: Tuple[str, str]) -> Optional[DialogueObject]:
        raise NotImplementedError("Must implement maybe_get_dialogue_object in subclass")
