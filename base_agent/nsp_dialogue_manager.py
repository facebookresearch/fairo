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
from time import time
from typing import Dict, Optional, List

from base_agent.memory_nodes import ProgramNode
from base_agent.dialogue_manager import DialogueManager
from base_agent.dialogue_objects import (
    BotGreet,
    DialogueObject,
    Say,
    coref_resolve,
    process_spans_and_remove_fixed_value,
)
from base_util import hash_user
from craftassist.test.validate_json import JSONValidator
from dialogue_model import SemanticParsingModel
from dlevent import sio
from nsp_logger import NSPLogger

spacy_model = spacy.load("en_core_web_sm")


class NSPDialogueManager(DialogueManager):
    """Dialogue manager driven by querying a model and
    doing an interpreter handover.

    Attributes:
        dialogue_objects (dict): Dictionary specifying the DialogueObject
            class for each dialogue type. Keys are dialogue types. Values are
            corresponding class names. Example dialogue objects:
            {'interpreter': MCInterpreter,
            'get_memory': GetMemoryHandler,
            'put_memory': ...
            }
        safety_words (Set(str)): Set of blacklisted words or phrases. Commands
            containing these are automatically filtered out.
        botGreetings (dict): Different types of greetings that trigger
            scripted responses. Example:
            { "hello": ["hi bot", "hello"] }
        model (TTADBertModel): Semantic Parsing model that takes text as
            input and outputs a logical form.
            To use a new model here, ensure that the subfolder directory structure
            mirrors the current model/dataset directories.
            See :class:`TTADBertModel`.
        ground_truth_actions (dict): A key-value with ground truth logical forms.
            These will be queried first (via exact string match), before running the model.
        dialogue_object_parameters (dict): Set the parameters for dialogue objects.
            Sets the agent, memory and dialogue stack.

    Args:
        agent: a droidlet agent, eg. ``CraftAssistAgent``
        dialogue_object_classes (dict): Dictionary specifying the DialogueObject
            class for each dialogue type. See ``dialogue_objects`` definition above.
        opts (argparse.Namespace): Parsed command line arguments specifying parameters in agent.

            Args:
                --nsp_models_dir: Path to directory containing all files necessary to
                    load and run the model, including args, tree mappings and the checkpointed model.
                    Semantic parsing models used by current project are in ``ttad_bert_updated``.
                    eg. semantic parsing model is ``ttad_bert_updated/caip_test_model.pth``
                --nsp_data_dir: Path to directory containing all datasets used by the NSP model.
                    Note that this data is not used in inference, rather we load from the ground truth
                    data directory.
                --ground_truth_data_dir: Path to directory containing ground truth datasets
                    loaded by agent at runtime. Option to include a file for blacklisted words ``safety.txt``,
                    a class for greetings ``greetings.json`` and .txt files with text, logical_form pairs in
                    ``datasets/``.

            See :class:`ArgumentParser` for full list of command line options.

    """

    def __init__(self, agent, dialogue_object_classes, opts):
        super(NSPDialogueManager, self).__init__(agent, None)
        # Read all datasets
        self.read_datasets(opts)
        self.NSPLogger = NSPLogger(
            "nsp_outputs.csv", ["command", "action_dict", "source", "agent", "time"]
        )
        self.dialogue_objects = dialogue_object_classes
        self.dialogue_object_parameters = {
            "agent": self.agent,
            "memory": self.agent.memory,
            "dialogue_stack": self.dialogue_stack,
        }
        # Instantiate the dialog model
        try:
            self.model = SemanticParsingModel(opts.nsp_models_dir, opts.nsp_data_dir)
        except NotADirectoryError:
            pass

        # Socket event listener
        @sio.on("queryParser")
        def query_parser(sid, data):
            """This is a socket event listener from dashboard and returns
            the logical form output"""
            logging.debug("inside query parser, querying for: %r" % (data))
            action_dict = self.get_logical_form(s=data["chat"], model=self.model)
            logging.debug("got logical form: %r" % (action_dict))
            payload = {"action_dict": action_dict}
            sio.emit("renderActionDict", payload)

    def read_datasets(self, opts):
        """Read all dataset files for DialogueManager:
        safety.txt, greetings.json, ground_truth/ folder
        """
        # Extract the set of safety words from safety file. This is used in the
        # step() method of DialogueManager when a chat is received.
        self.safety_words = set()
        safety_words_path = opts.ground_truth_data_dir + "safety.txt"
        if os.path.isfile(safety_words_path):
            self.safety_words = self.get_safety_words(safety_words_path)

        # Load greetings
        greetings_path = opts.ground_truth_data_dir + "greetings.json"
        self.botGreetings = {"hello": ["hi", "hello", "hey"], "goodbye": ["bye"]}
        if os.path.isfile(greetings_path):
            with open(greetings_path) as fd:
                self.botGreetings = json.load(fd)

        # Load all ground truth commands and parses
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

    def maybe_get_dialogue_obj(
        self, speaker: str, chat_list: List[str]
    ) -> Optional[DialogueObject]:
        """This is the function that is called from the step() of DialogueManager.
        This function processes a chat and modified the dialogue stack if
        necessary.
        The order is:
        1. check againt greetings and return
        2. check against ground_truth commands or query model
        3. handle the logical form by first: processing spans + resolving coreference
        and then handing over to interpreter.

        Args:
            chat str: Incoming chat from a player.
                Format is chat, eg. "build a red house"

        Returns:
            DialogueObject or empty if no action is needed.
        """
        # NOTE:At the moment, we are only handling one chat at a
        # time but this is up for discussion.
        chat = chat_list[0]

        # If we are waiting on a response from the user (e.g.: an answer
        # to a clarification question asked), return None.
        if (len(self.dialogue_stack) > 0) and (self.dialogue_stack[-1].awaiting_response):
            return None

        # 1. Check if incoming chat is one of the scripted ones in greetings
        # and push appropriate DialogueObjects to stack.
        for greeting_type, allowed_str in self.botGreetings.items():
            if chat in allowed_str:
                return BotGreet(greeting_type, **self.dialogue_object_parameters)

        # 2. Get logical form from either ground truth or query the model
        logical_form = self.get_logical_form(chat=chat, model=self.model)
        # 3. return the DialogueObject
        return self.handle_logical_form(speaker, logical_form, chat)

    def validate_parse_tree(self, parse_tree: Dict) -> bool:
        """Validate the parse tree against current grammar."""
        # RefResolver initialization requires a base schema and URI
        schema_dir = "{}/".format(
            pkg_resources.resource_filename("base_agent.documents", "json_schema")
        )
        json_validator = JSONValidator(schema_dir, span_type="all")
        is_valid_json = json_validator.validate_instance(parse_tree)
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
        # updates are made to the dictionary in-place
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

    def get_logical_form(self, chat: str, model) -> Dict:
        """Get logical form output for a given chat command.
        First check the ground truth file for the chat string. If not
        in ground truth, query semantic parsing model to get the output.

        Args:
            chat (str): Input chat provided by the user.
            model (TTADBertModel): Semantic parsing model, pre-trained and loaded
                by agent

        Return:
            Dict: Logical form representation of the task. See paper for more
                in depth explanation of logical forms:
                https://arxiv.org/abs/1907.08584

        Examples:
            >>> get_logical_form("destroy this", model)
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
        # Check if chat is in ground_truth otherwise query model
        if chat in self.ground_truth_actions:
            logical_form = copy.deepcopy(self.ground_truth_actions[chat])
            logging.info('Found ground truth action for "{}"'.format(chat))
            # log the current UTC time
            time_now = time()
        else:
            logical_form = model.get_logical_form(chat)
            time_now = time()
            logical_form_source = "semantic_parser"
        # log the logical form with source
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
        # First postprocess logical form: fill spans + resolve coreference
        logical_form = self.postprocess_logical_form(speaker, chat, logical_form)
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
