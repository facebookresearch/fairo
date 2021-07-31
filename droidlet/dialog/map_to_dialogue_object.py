from enum import Enum
import json
import logging
import random
import re
import sentry_sdk
import spacy
from typing import Dict, Optional
from droidlet.dialog.dialogue_objects import DialogueObject, Say
from droidlet.interpreter import coref_resolve, process_spans_and_remove_fixed_value
from droidlet.base_util import hash_user
from .load_datasets import get_greetings, get_safety_words

spacy_model = spacy.load("en_core_web_sm")


class GreetingType(Enum):
    """Types of bot greetings."""

    HELLO = "hello"
    GOODBYE = "goodbye"


class DialogueObjectMapper(object):
    def __init__(self,
                 dialogue_object_classes,
                 opts,
                 low_level_interpreter_data,
                 dialogue_manager):
        self.dialogue_objects = dialogue_object_classes
        self.opts = opts
        self.low_level_interpreter_data = low_level_interpreter_data
        self.dialogue_manager = dialogue_manager
        self.safety_words = get_safety_words()
        self.greetings = get_greetings(self.opts.ground_truth_data_dir)

    def get_dialogue_object(self, speaker: str, chat: str, parse: Dict, chat_status: str, chat_memid: str):
        """Returns DialogueObject for a given chat and logical form"""
        # 1. If we are waiting on a response from the user (e.g.: an answer
        # to a clarification question asked), return None.
        if (len(self.dialogue_manager.dialogue_stack) > 0) and (
                self.dialogue_manager.dialogue_stack[-1].awaiting_response
        ):
            return None

        # If chat has been processed already, return
        if not chat_status:
            return None
        # Mark chat as processed
        self.dialogue_manager.memory.untag(chat_memid, "unprocessed")

        # 1. Check against safety phrase list
        if not self.is_safe(chat):
            return Say("Please don't be rude.", memory=self.dialogue_manager.memory)

        # 2. Check if incoming chat is one of the scripted ones in greetings
        reply = self.get_greeting_reply(chat)
        if reply:
            return Say(reply, memory=self.dialogue_manager.memory)

        # 3. postprocess logical form: process spans + resolve coreference
        updated_logical_form = self.postprocess_logical_form(
            speaker=speaker, chat=chat, logical_form=parse
        )

        # 4. handle the logical form by returning appropriate DialogueObject.
        return self.handle_logical_form(
            speaker=speaker, logical_form=updated_logical_form, chat=chat, opts=self.opts
        )

    def postprocess_logical_form(self, speaker: str, chat: str, logical_form: Dict) -> Dict:
        """This function performs some postprocessing on the logical form:
        substitutes indices with words and resolves coreference"""
        # perform lemmatization on the chat
        logging.debug('chat before lemmatization "{}"'.format(chat))
        lemmatized_chat = spacy_model(chat)
        lemmatized_chat_str = " ".join(str(word.lemma_) for word in lemmatized_chat)
        logging.debug('chat after lemmatization "{}"'.format(lemmatized_chat_str))

        # Get the words from indices in spans and substitute fixed_values
        # NOTE: updates are made to the dictionary in-place
        process_spans_and_remove_fixed_value(
            logical_form, re.split(r" +", chat), re.split(r" +", lemmatized_chat_str)
        )

        # log to sentry
        sentry_sdk.capture_message(
            json.dumps({"type": "ttad_pre_coref", "in_original": chat, "out": logical_form})
        )
        sentry_sdk.capture_message(
            json.dumps(
                {
                    "type": "ttad_pre_coref",
                    "in_lemmatized": lemmatized_chat_str,
                    "out": logical_form,
                }
            )
        )
        logging.debug('ttad pre-coref "{}" -> {}'.format(lemmatized_chat_str, logical_form))

        # Resolve any co-references like "this", "that" "there" using heuristics
        # and make updates in the dictionary in place.
        coref_resolve(self.dialogue_manager.memory, logical_form, chat)
        logging.debug(
            'logical form post co-ref "{}" -> {}'.format(hash_user(speaker), logical_form)
        )
        return logical_form

    def handle_logical_form(
        self, speaker: str, logical_form: Dict, chat: str, opts=None
    ) -> Optional[DialogueObject]:
        """Return the appropriate DialogueObject to handle an action dict d
        d should have spans filled (via process_spans_and_remove_fixed_value).
        """
        memory = self.dialogue_manager.memory
        if logical_form["dialogue_type"] == "NOOP":
            return Say("I don't know how to answer that.", memory=memory)
        elif logical_form["dialogue_type"] == "GET_CAPABILITIES":
            return self.dialogue_objects["bot_capabilities"](memory=memory)
        elif logical_form["dialogue_type"] == "HUMAN_GIVE_COMMAND":
            # _BIG_ FIXME: self.low_level_interpreter_data should be removed
            return self.dialogue_objects["interpreter"](
                speaker, logical_form, self.low_level_interpreter_data, memory=memory
            )
        elif logical_form["dialogue_type"] == "PUT_MEMORY":
            return self.dialogue_objects["put_memory"](speaker, logical_form, memory=memory)
        elif logical_form["dialogue_type"] == "GET_MEMORY":
            return self.dialogue_objects["get_memory"](speaker, logical_form, memory=memory)
        else:
            raise ValueError("Bad dialogue_type={}".format(logical_form["dialogue_type"]))

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


