"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from typing import Tuple, Dict


class DialogueManager(object):
    """
    | The current control flow of dialogue is:
    | 1.  A chat comes in and Dialogue manager reads it or the bot triggers a
    |    dialogue because of memory/perception/task state
    | 2.  The dialogue manage launches an Interpreter or places a dialogue Task on the Task queue.
    | 3.  the agent calls the Interpreters.step() (if there is an Interpreter one) in its
    |    controller_step()
    |
    | This object is likely to be deprecated by nov 2021


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
            logical_form_memid, chat_status = None, ""
            logical_form_triples = self.memory.get_triples(
                subj=chat_memid, pred_text="has_logical_form"
            )
            processed_status = self.memory.get_triples(
                subj=chat_memid, pred_text="has_tag", obj_text="uninterpreted"
            )
            if logical_form_triples:
                logical_form_memid = logical_form_triples[0][2]

            if processed_status:
                chat_status = processed_status[0][2]
            chat_str = chat.chat_text
            chat_list_text.append((speaker, chat_str, logical_form_memid, chat_status, chat_memid))

        return chat_list_text

    def step(self):
        """Process a chat and step through the dialogue manager task stack.

        The chat is given as input to the model, which returns a logical form.
        The logical form is fed to an interpreter to
        handle the action.

        Args:
            chat (Tuple[str, str]): Tuple of (speaker, chat). Speaker is name of speaker.
                Chat is an incoming chat command that the agent has received.
                Example: ("player_1", "build a red cube")

        """
        # chat is a single line command
        chat_list = self.get_last_m_chats(m=1)
        # TODO: this can be moved to get_d_o

        if chat_list:
            # TODO: remove this and have mapper take in full list
            speaker, chatstr, logical_form_memid, chat_status, chat_memid = chat_list[0]

            # NOTE: the model is responsible for not putting a new
            # object on the stack if it sees that whatever is on
            # the stack should continue.
            # TODO: Change this to only take parse and use get_last_m_chats to get chat + speaker
            return self.dialogue_object_mapper.get_dialogue_object(
                speaker, chatstr, logical_form_memid, chat_status, chat_memid
            )
