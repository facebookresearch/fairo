"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import copy
import json
import logging
import os
import pkg_resources
import re
import sentry_sdk
import spacy
from droidlet.dialog import preprocess
from time import time
from typing import Dict, Optional
from droidlet.dialog.semantic_parser_wrapper import SemanticParserWrapper
from droidlet.memory.memory_nodes import ProgramNode

from droidlet.dialog.dialogue_objects import (
    BotGreet,
    DialogueObject,
    Say,
    coref_resolve,
    process_spans_and_remove_fixed_value,
)
from droidlet.shared_data_struct.base_util import hash_user

# TODO: move JSONValidator into base
from droidlet.dialog.craftassist.tests.validate_json import JSONValidator
from droidlet.dialog.dialogue_model import DroidletSemanticParsingModel
from droidlet.event import sio
from droidlet.dialog.nsp_logger import NSPLogger

spacy_model = spacy.load("en_core_web_sm")


class DroidletNSPModelWrapper(SemanticParserWrapper):
    def __init__(self, agent, dialogue_object_classes, opts, dialogue_manager):
        super(DroidletNSPModelWrapper, self).__init__(
            agent, dialogue_object_classes, opts, dialogue_manager
        )
        # Read all datasets
        self.read_datasets(opts)
        # instantiate logger and parsing model
        self.NSPLogger = NSPLogger(
            "nsp_outputs.csv", ["command", "action_dict", "source", "agent", "time"]
        )
        try:
            self.parsing_model = DroidletSemanticParsingModel(
                opts.nsp_models_dir, opts.nsp_data_dir
            )
        except NotADirectoryError:
            # No parsing model
            self.parsing_model = None

        # Socket event listener
        # TODO(kavyas): I might want to move this to SemanticParserWrapper
        @sio.on("queryParser")
        def query_parser(sid, data):
            """This is a socket event listener from dashboard and returns
            the logical form output"""
            logging.debug("inside query parser, querying for: %r" % (data))
            action_dict = self.get_logical_form(s=data["chat"], parsing_model=self.parsing_model)
            logging.debug("got logical form: %r" % (action_dict))
            payload = {"action_dict": action_dict}
            sio.emit("renderActionDict", payload)

    def read_datasets(self, opts):
        """Read all dataset files for DialogueManager:
        safety.txt, greetings.json, ground_truth/datasets folder
        """
        # Extract the set of safety words from safety file
        self.safety_words = set()
        safety_words_path = "{}/{}".format(
            pkg_resources.resource_filename("droidlet.documents", "internal"),
            "safety.txt",
        )
        if os.path.isfile(safety_words_path):
            """Read a set of safety words to prevent abuse."""
            with open(safety_words_path) as f:
                for l in f.readlines():
                    w = l.strip("\n").lower()
                    if w != "" and w[0] != "<" and w[0] != "#":
                        self.safety_words.add(w)

        # Load greetings
        greetings_path = opts.ground_truth_data_dir + "greetings.json"
        self.botGreetings = {"hello": ["hi", "hello", "hey"], "goodbye": ["bye"]}
        if os.path.isfile(greetings_path):
            with open(greetings_path) as fd:
                self.botGreetings = json.load(fd)

        # Load all ground truth commands and their parses
        self.ground_truth_actions = {}
        if not opts.no_ground_truth:
            if os.path.isdir(opts.ground_truth_data_dir):
                gt_data_directory = opts.ground_truth_data_dir + "datasets/"
                for (dirpath, dirnames, filenames) in os.walk(gt_data_directory):
                    for f_name in filenames:
                        file = gt_data_directory + f_name
                        with open(file) as f:
                            for line in f.readlines():
                                text, logical_form = line.strip().split("|")
                                clean_text = text.strip('"')
                                self.ground_truth_actions[clean_text] = json.loads(logical_form)

    def is_safe(self, chat):
        """Check that chat does not contain any word from the
        safety check list.
        """
        cmd_set = set(chat.lower().split())
        notsafe = len(cmd_set & self.safety_words) > 0
        return not notsafe

    def preprocess_chat(self, chat):
        """Tokenize the chat and get list of sentences to parse.
        This can be swapped out with another preprocessor.
        """
        preprocessed_chat = preprocess.preprocess_chat(chat)
        return preprocessed_chat

    def get_dialogue_object(self) -> Optional[DialogueObject]:
        """This is the function that is called from the step() of DialogueManager.
        This function processes a chat and modified the dialogue stack if
        necessary.
        The order is:
        1. Check if pending confirmation
        2. Preprocess chat
        3. check against safety words first
        4. check againt greetings and return
        5. check against ground_truth commands or query model
        6. postprocess the logical form: processing spans + resolving coreference
        7. handle the logical form by returning appropriate DialogueObject.


        Returns:
            DialogueObject or empty if no action is needed.
        """
        # 1. If we are waiting on a response from the user (e.g.: an answer
        # to a clarification question asked), return None.
        if (len(self.dialogue_manager.dialogue_stack) > 0) and (
            self.dialogue_manager.dialogue_stack[-1].awaiting_response
        ):
            return None

        # NOTE: We are only handling the last chat here compared to full chat history
        chat_list = self.dialogue_manager.get_last_m_chats(m=1)

        # 2. Preprocess chat
        speaker, chatstr = chat_list[0]
        chat = self.preprocess_chat(chatstr)

        # 3. Check against safety phrase list
        if not self.is_safe(chat):
            return Say("Please don't be rude.", **self.dialogue_object_parameters)

        # 4. Check if incoming chat is one of the scripted ones in greetings
        for greeting_type, allowed_str in self.botGreetings.items():
            if chat in allowed_str:
                return BotGreet(greeting_type, **self.dialogue_object_parameters)

        # 5. Get logical form from either ground truth or query the parsing model
        logical_form = self.get_logical_form(chat=chat, parsing_model=self.parsing_model)

        # 6. postprocess logical form: fill spans + resolve coreference
        updated_logical_form = self.postprocess_logical_form(
            speaker=speaker, chat=chat, logical_form=logical_form
        )

        # 7. return the DialogueObject
        return self.handle_logical_form(
            speaker=speaker, logical_form=updated_logical_form, chat=chat
        )

    def validate_parse_tree(self, parse_tree: Dict, debug: bool = True) -> bool:
        """Validate the parse tree against current grammar.
        
        Args:
            parse_tree (Dict): logical form to be validated.
            debug (bool): whether to print error trace for debugging.

        Returns:
            True if parse tree is valid, False if not.
        """
        # RefResolver initialization requires a base schema and URI
        schema_dir = "{}/".format(
            pkg_resources.resource_filename("droidlet.documents", "json_schema")
        )
        json_validator = JSONValidator(schema_dir, span_type="all")
        is_valid_json = json_validator.validate_instance(parse_tree, debug)
        return is_valid_json

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

        # Resolve any coreferences like "this", "that" "there" using heuristics
        # and make updates in the dictionary in place.
        coref_resolve(self.agent.memory, logical_form, chat)
        logging.debug(
            'logical form post-coref "{}" -> {}'.format(hash_user(speaker), logical_form)
        )

        return logical_form

    def get_logical_form(self, chat: str, parsing_model) -> Dict:
        """Get logical form output for a given chat command.
        First check the ground truth file for the chat string. If not
        in ground truth, query semantic parsing model to get the output.

        Args:
            chat (str): Input chat provided by the user.
            parsing_model (TTADBertModel): Semantic parsing model, pre-trained and loaded
                by agent

        Return:
            Dict: Logical form representation of the task. See paper for more
                in depth explanation of logical forms:
                https://arxiv.org/abs/1907.08584

        Examples:
            >>> get_logical_form("destroy this", parsing_model)
            {
                "dialogue_type": "HUMAN_GIVE_COMMAND",
                "action_sequence": [{
                    "action_type": "DESTROY",
                    "reference_object": {
                        "filters": {"contains_coreference": "yes"},
                        "text_span": [0, [1, 1]]
                    }
                }]
            }
        """
        logical_form_source = "ground_truth"
        # Check if chat is in ground_truth otherwise query parsing model
        if chat in self.ground_truth_actions:
            logical_form = copy.deepcopy(self.ground_truth_actions[chat])
            logging.info('Found ground truth action for "{}"'.format(chat))
            # log the current UTC time
            time_now = time()
        elif self.parsing_model:
            logical_form = parsing_model.query_for_logical_form(chat)
            time_now = time()
            logical_form_source = "semantic_parser"
        else:
            logical_form = {"dialogue_type": "NOOP"}
            logging.info("Not found in ground truth, no parsing model initiated. Returning NOOP.")
            time_now = time()
            logical_form_source = "not_found_in_gt_no_model"
        # log the logical form and chat with source
        self.NSPLogger.log_dialogue_outputs(
            [chat, logical_form, logical_form_source, "craftassist", time_now]
        )
        # check if logical_form conforms to the grammar
        is_valid_json = self.validate_parse_tree(logical_form)
        if not is_valid_json:
            # Send a NOOP
            logging.error("Invalid parse tree for command %r \n" % (chat))
            logging.error("Parse tree failed grammar validation: \n %r \n" % (logical_form))
            logical_form = {"dialogue_type": "NOOP"}
            logging.error("Returning NOOP")

        return logical_form

    def handle_logical_form(
        self, speaker: str, logical_form: Dict, chat: str
    ) -> Optional[DialogueObject]:
        """Return the appropriate DialogueObject to handle an action dict d
        d should have spans filled (via process_spans_and_remove_fixed_value).
        """
        ProgramNode.create(self.agent.memory, logical_form)

        if logical_form["dialogue_type"] == "NOOP":
            return Say("I don't know how to answer that.", **self.dialogue_object_parameters)
        elif logical_form["dialogue_type"] == "GET_CAPABILITIES":
            return self.dialogue_objects["bot_capabilities"](**self.dialogue_object_parameters)
        elif logical_form["dialogue_type"] == "HUMAN_GIVE_COMMAND":
            return self.dialogue_objects["interpreter"](
                speaker, logical_form, **self.dialogue_object_parameters
            )
        elif logical_form["dialogue_type"] == "PUT_MEMORY":
            return self.dialogue_objects["put_memory"](
                speaker, logical_form, **self.dialogue_object_parameters
            )
        elif logical_form["dialogue_type"] == "GET_MEMORY":
            return self.dialogue_objects["get_memory"](
                speaker, logical_form, **self.dialogue_object_parameters
            )
        else:
            raise ValueError("Bad dialogue_type={}".format(logical_form["dialogue_type"]))
