"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from typing import Tuple, Optional
from .dialogue_stack import DialogueStack
from .dialogue_objects import DialogueObject


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
        
    """

    def __init__(
        self, agent, dialogue_object_classes, opts, semantic_parsing_model_wrapper
    ):
        self.agent = agent
        self.dialogue_stack = DialogueStack(agent, agent.memory)
        self.semantic_parsing_model_wrapper = semantic_parsing_model_wrapper(
            agent=self.agent,
            dialogue_object_classes=dialogue_object_classes,
            opts=opts,
            dialogue_manager=self,
        )

    def get_last_m_chats(self, m=1):
        # fetch last m chats from memory
        all_chats = self.agent.memory.get_recent_chats(n=m)
        chat_list_text = []
        for chat in all_chats:
            speaker = self.agent.memory.get_player_by_id(chat.speaker_id).name
            chat_str = chat.chat_text
            chat_list_text.append((speaker, chat_str))
        
        return chat_list_text

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

        if chatstr:
            logging.debug("Dialogue stack pre-run_model: {}".format(self.dialogue_stack.stack))

            # NOTE: the model is responsible for not putting a new
            # object on the stack if it sees that whatever is on
            # the stack should continue.
            # TODO: Maybe we need a HoldOn dialogue object?
            obj = self.semantic_parsing_model_wrapper.get_dialogue_object()
            if obj is not None:
                self.dialogue_stack.append(obj)

        # Always call dialogue_stack.step(), even if chat is empty
        if len(self.dialogue_stack) > 0:
            self.dialogue_stack.step()
