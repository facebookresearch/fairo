"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import datetime
from droidlet.event import dispatch
from typing import Tuple, Dict


class DialogueManager(object):
    """
    | The current control flow of dialogue is:
    | 1.  A chat comes in and Dialogue manager reads it or the bot triggers a
    |    dialogue because of memory/perception/task state
    | 2.  The dialogue manager puts a DialogueObject on the DialogueStack.
    |    or dialogue Task on the Task queue.  Near future: DialogueStack will be deprecated
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
        self,
        memory,
        dialogue_object_classes,
        dialogue_object_mapper,
        opts,
        low_level_interpreter_data={},
    ):
        self.memory = memory
        # FIXME in stage III; need a sensible interface for this
        self.dialogue_stack = memory.dialogue_stack
        self.dialogue_object_mapper = dialogue_object_mapper(
            dialogue_object_classes=dialogue_object_classes,
            opts=opts,
            low_level_interpreter_data=low_level_interpreter_data,
            dialogue_manager=self,
        )

    def get_last_m_chats(self, m=1):
        # fetch last m chats from memory
        all_chats = self.memory.get_recent_chats(n=m)
        chat_list_text = []
        for chat in all_chats:
            speaker = self.memory.get_player_by_id(chat.speaker_id).name
            chat_memid = chat.memid
            # get logical form if any else None
            logical_form, chat_status = None, ""
            logical_form_triples = self.memory.get_triples(
                subj=chat_memid, pred_text="has_logical_form"
            )
            processed_status = self.memory.get_triples(
                subj=chat_memid, pred_text="has_tag", obj_text="unprocessed"
            )
            if logical_form_triples:
                logical_form = self.memory.get_logical_form_by_id(
                    logical_form_triples[0][2]
                ).logical_form

            if processed_status:
                chat_status = processed_status[0][2]
            chat_str = chat.chat_text
            chat_list_text.append((speaker, chat_str, logical_form, chat_status, chat_memid))

        return chat_list_text

    def step(self):
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
        start_time = datetime.datetime.now()
        # chat is a single line command
        chat_list = self.get_last_m_chats(m=1)
        # TODO: this can be moved to get_d_o

        if chat_list:
            # TODO: remove this and have mapper take in full list
            speaker, chatstr, logical_form, chat_status, chat_memid = chat_list[0]
            logging.debug("Dialogue stack pre-run_model: {}".format(self.dialogue_stack.stack))

            # NOTE: the model is responsible for not putting a new
            # object on the stack if it sees that whatever is on
            # the stack should continue.
            # TODO: Maybe we need a HoldOn dialogue object?
            # TODO: Change this to only take parse and use get_last_m_chats to get chat + speaker
            obj = self.dialogue_object_mapper.get_dialogue_object(
                speaker, chatstr, logical_form, chat_status, chat_memid
            )
            if obj is not None:
                if type(obj) is not dict:
                    self.dialogue_stack.append(obj)
                end_time = datetime.datetime.now()
                hook_data = {
                    "name": "dialogue",
                    "start_time": start_time,
                    "end_time": end_time,
                    "elapsed_time": (end_time - start_time).total_seconds(),
                    "agent_time": self.memory.get_time(),
                    "object": str(obj),
                }
                dispatch.send("dialogue", data=hook_data)
                return obj
