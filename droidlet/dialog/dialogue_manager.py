"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from enum import Enum
import random
from typing import Tuple

from droidlet.dialog.dialogue_task import Say
from droidlet.memory.memory_nodes import ChatNode, PlayerNode, TripleNode
from .load_datasets import get_greetings, get_safety_words


class GreetingType(Enum):
    """Types of bot greetings."""

    HELLO = "hello"
    GOODBYE = "goodbye"


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
        opts,
        low_level_interpreter_data={},
    ):
        self.memory = memory
        self.dialogue_objects = dialogue_object_classes
        self.opts = opts
        self.low_level_interpreter_data = low_level_interpreter_data
        self.safety_words = get_safety_words()
        self.greetings = get_greetings(self.opts.ground_truth_data_dir)

    def get_last_m_chats(self, m=1):
        # fetch last m chats from memory
        all_chats = self.memory.nodes[ChatNode.NODE_TYPE].get_recent_chats(self.memory, n=m)
        chat_list_text = []
        for chat in all_chats:
            speaker = self.memory.nodes[PlayerNode.NODE_TYPE](self.memory, chat.speaker_id).name
            chat_memid = chat.memid
            # get logical form if any else None
            logical_form_memid, chat_status = None, ""
            logical_form_triples = self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                self.memory, subj=chat_memid, pred_text="has_logical_form"
            )
            processed_status = self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                self.memory, subj=chat_memid, pred_text="has_tag", obj_text="uninterpreted"
            )
            if logical_form_triples:
                logical_form_memid = logical_form_triples[0][2]

            if processed_status:
                chat_status = processed_status[0][2]
            chat_str = chat.chat_text
            chat_list_text.append((speaker, chat_str, logical_form_memid, chat_status, chat_memid))

        return chat_list_text

    def get_dialogue_object(
        self, speaker: str, chat: str, logical_form_memid: str, chat_status: str, chat_memid: str
    ):
        """Returns DialogueObject (or ingredients for a DialogueTask)
        for a given chat and logical form"""

        # 1. If we are waiting on a response from the user (e.g.: an answer
        # to a clarification question asked), return None.

        # for now if any interpreter is active, it takes precedence.  eventually use Task queue to manage
        _, interpreter_mems = self.memory.basic_search(
            "SELECT MEMORY FROM Interpreter WHERE finished = 0"
        )
        if len(interpreter_mems) > 0:  # TODO temporary error if >1?
            mem = interpreter_mems[0]
            cls = self.dialogue_objects.get(mem.interpreter_type)
            if cls is not None:
                return cls(
                    speaker,
                    logical_form_memid,
                    self.memory,
                    memid=mem.memid,
                    low_level_data=self.low_level_interpreter_data,
                )
            else:
                raise Exception(
                    "tried to build unknown intepreter type from memory {}".format(
                        mem.interpreter_type
                    )
                )

        _, active_task_mems = self.memory.basic_search("SELECT MEMORY FROM Task WHERE prio > -1")
        dialogue_task_busy = any(
            [getattr(m, "awaiting_response", False) for m in active_task_mems]
        )
        if dialogue_task_busy:
            return None

        # If chat has been processed already, return
        if not chat_status:
            return None
        # Mark chat as processed
        self.memory.nodes[TripleNode.NODE_TYPE].untag(self.memory, chat_memid, "uninterpreted")

        # FIXME handle this in gt (all of this will be folded into manager
        # 1. Check against safety phrase list
        if not self.is_safe(chat):
            return {"task": Say, "data": {"response_options": "Please don't be rude."}}

        # FIXME handle this in gt (all of this will be folded into manager
        # 2. Check if incoming chat is one of the scripted ones in greetings
        reply = self.get_greeting_reply(chat)
        if reply:
            return {"task": Say, "data": {"response_options": reply}}

        # 3. handle the logical form by returning appropriate Interpreter or dialogue task.
        return self.handle_logical_form(speaker, logical_form_memid)

    def handle_logical_form(self, speaker, logical_form_memid):
        """Return the appropriate interpreter to handle a logical form in memory
        the logical form should have spans filled (via process_spans_and_remove_fixed_value).
        """
        memory = self.memory
        logical_form = memory.get_mem_by_id(logical_form_memid).logical_form

        if logical_form["dialogue_type"] == "NOOP":
            return {"task": Say, "data": {"response_options": "I don't know how to answer that."}}
        elif logical_form["dialogue_type"] == "GET_CAPABILITIES":
            return self.dialogue_objects["bot_capabilities"]
        else:
            # _BIG_ FIXME: self.low_level_interpreter_data should be removed
            I = {
                "HUMAN_GIVE_COMMAND": self.dialogue_objects["interpreter"],
                "PUT_MEMORY": self.dialogue_objects["put_memory"],
                "GET_MEMORY": self.dialogue_objects["get_memory"],
            }.get(logical_form["dialogue_type"])
            if not I:
                raise ValueError("Bad dialogue_type={}".format(logical_form["dialogue_type"]))
            return I(
                speaker, logical_form_memid, memory, low_level_data=self.low_level_interpreter_data
            )

    def is_safe(self, chat):
        """Check that chat does not contain any word from the
        safety check list.
        """
        cmd_set = set(chat.lower().split())
        notsafe = len(cmd_set & self.safety_words) > 0
        return not notsafe

    def get_greeting_reply(self, chat):
        response_options = []
        for greeting_type, allowed_str in self.greetings.items():
            if chat in allowed_str:
                if greeting_type == GreetingType.GOODBYE.value:
                    response_options = ["goodbye", "bye", "see you next time!"]
                else:
                    response_options = ["hi there!", "hello", "hey", "hi"]
                return random.choice(response_options)
        return None

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
            return self.get_dialogue_object(
                speaker, chatstr, logical_form_memid, chat_status, chat_memid
            )
