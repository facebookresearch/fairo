import json
import logging
import re
import sentry_sdk
import spacy
from typing import Dict, Optional
from droidlet.memory.memory_nodes import ProgramNode
from droidlet.dialog.dialogue_objects import DialogueObject, Say
from droidlet.interpreter import coref_resolve, process_spans_and_remove_fixed_value
from droidlet.base_util import hash_user

spacy_model = spacy.load("en_core_web_sm")


class DialogueObjectMapper(object):
    def __init__(self, dialogue_object_classes, opts, dialogue_manager):
        self.dialogue_objects = dialogue_object_classes
        self.opts = opts
        self.dialogue_manager = dialogue_manager

    def get_dialogue_object(self, speaker: str, chat: str, parse: str or Dict):
        # # NOTE: We are only handling the last chat here compared to full chat history
        # chat_list = self.dialogue_manager.get_last_m_chats(m=1)
        #
        # # 2. Preprocess chat
        # speaker, chatstr = chat_list[0]
        # chat = self.preprocess_chat(chatstr)

        # 1. If we are waiting on a response from the user (e.g.: an answer
        # to a clarification question asked), return None.
        if (len(self.dialogue_manager.dialogue_stack) > 0) and (
                self.dialogue_manager.dialogue_stack[-1].awaiting_response
        ):
            return None

        """
        2. Check if parse was string or logical form. If string, the agent will
        reply back to player with the string
        """
        # Note make `Say` only take in a string and return that.
        if type(parse) == str:
            return Say(parse, memory=self.dialogue_manager.memory)

        """
        Else the parse is a logical form and we :
        3. postprocess the logical form: processing spans + resolving coreference
        4. handle the logical form by returning appropriate DialogueObject.
        """
        # 3. postprocess logical form: fill spans + resolve coreference
        updated_logical_form = self.postprocess_logical_form(
            speaker=speaker, chat=chat, logical_form=parse
        )

        # 4. return the DialogueObject
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
        ProgramNode.create(self.dialogue_manager.memory, logical_form)
        memory = self.dialogue_manager.memory
        if logical_form["dialogue_type"] == "NOOP":
            return Say("I don't know how to answer that.", memory=memory)
        elif logical_form["dialogue_type"] == "GET_CAPABILITIES":
            return self.dialogue_objects["bot_capabilities"](memory=memory)
        elif logical_form["dialogue_type"] == "HUMAN_GIVE_COMMAND":
            low_level_interpreter_data = {}
            if opts and hasattr(opts, 'block_data'):
                low_level_interpreter_data['block_data'] = opts.block_data
            if opts and hasattr(opts, 'special_shape_functions'):
                low_level_interpreter_data['special_shape_functions'] = opts.special_shape_functions
            return self.dialogue_objects["interpreter"](
                speaker, logical_form, low_level_interpreter_data, memory=memory
            )
        elif logical_form["dialogue_type"] == "PUT_MEMORY":
            return self.dialogue_objects["put_memory"](speaker, logical_form, memory=memory)
        elif logical_form["dialogue_type"] == "GET_MEMORY":
            return self.dialogue_objects["get_memory"](speaker, logical_form, memory=memory)
        else:
            raise ValueError("Bad dialogue_type={}".format(logical_form["dialogue_type"]))
