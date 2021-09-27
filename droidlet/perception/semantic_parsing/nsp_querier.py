"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import copy
import logging
import pkg_resources
import re
import time
from typing import Dict, Tuple
from .utils import preprocess
from .load_and_check_datasets import get_ground_truth
from .nsp_model_wrapper import DroidletSemanticParsingModel
from droidlet.event import sio
from .utils.nsp_logger import NSPLogger
from .utils.validate_json import JSONValidator
from droidlet.base_util import hash_user


class NSPQuerier(object):
    def __init__(self, opts, agent=None):
        """This class provides an API that takes in chat as plain text
         and converts it to logical form. It does so by first checking against
         ground truth text-logical form pairings and if not found, querying the
         neural semantic parsing model.
         """
        self.agent = agent
        self.opts = opts
        # instantiate logger and parsing model
        self.NSPLogger = NSPLogger(
            "nsp_outputs.csv", ["command", "action_dict", "source", "agent", "time"]
        )
        self.ErrorLogger = NSPLogger(
            "error_details.csv", ["command", "action_dict", "time", "parser_error", "other_error", "other_error_description"]
        )
        try:
            self.parsing_model = DroidletSemanticParsingModel(
                opts.nsp_models_dir, opts.nsp_data_dir
            )
        except NotADirectoryError:
            # No parsing model
            self.parsing_model = None
        # Read the ground truth dataset file: ground_truth/datasets folder
        self.ground_truth_actions = get_ground_truth(self.opts.no_ground_truth, self.opts.ground_truth_data_dir)

        # Socket event listener
        # TODO(kavya): I might want to move this to SemanticParserWrapper
        @sio.on("queryParser")
        def query_parser(sid, data):
            """This is a socket event listener from dashboard and returns
            the logical form output"""
            logging.debug("inside query parser, querying for: %r" % (data))
            action_dict = self.get_logical_form(s=data["chat"], parsing_model=self.parsing_model)
            logging.debug("got logical form: %r" % (action_dict))
            payload = {"action_dict": action_dict}
            sio.emit("renderActionDict", payload)

        @sio.on("saveErrorDetailsToCSV")
        def save_error_details(sid, data):
            """Save error details to error logs.
            The fields are
                ["command", "action_dict", "source", "agent", "time", "parser_error", "other_error", "other_error_description"]
            """
            logging.info("Saving error details: %r" % (data))
            if "action_dict" not in data or "msg" not in data:
                logging.info("Could not save error details due to error in dashboard backend.")
                return
            is_parser_error = data["parsing_error"]
            if is_parser_error:
                self.ErrorLogger.log_dialogue_outputs([data["msg"], data["action_dict"], None, True, None, None])
            else:
                self.ErrorLogger.log_dialogue_outputs([data["msg"], data["action_dict"], None, False, True, data["feedback"]])

    def perceive(self, force=False):
        """Get the incoming chats, preprocess the chat, run through the parser
        and return
        Process incoming chats and run through parser"""
        received_chats_flag = False
        speaker, chat, preprocessed_chat, chat_parse = "", "", "", {}
        raw_incoming_chats = self.agent.get_incoming_chats()
        if raw_incoming_chats:
            logging.info("Incoming chats: {}".format(raw_incoming_chats))
        incoming_chats = []
        for raw_chat in raw_incoming_chats:
            match = re.search("^<([^>]+)> (.*)", raw_chat)
            if match is None:
                logging.debug("Ignoring chat in NLU preceive: {}".format(raw_chat))
                continue

            speaker, chat = match.group(1), match.group(2)
            speaker_hash = hash_user(speaker)
            logging.debug("In NLU perceive, incoming chat: ['{}' -> {}]".format(speaker_hash, chat))
            if chat.startswith("/"):
                continue
            incoming_chats.append((speaker, chat))

        if len(incoming_chats) > 0:
            # force to get objects, speaker info
            if self.agent.perceive_on_chat:
                force = True
            self.agent.last_chat_time = time.time()
            # For now just process the first incoming chat, where chat -> [speaker, chat]
            speaker, chat = incoming_chats[0]
            received_chats_flag = True
            preprocessed_chat, chat_parse = self.get_parse(chat)

        return force, received_chats_flag, speaker, chat, preprocessed_chat, chat_parse

    def preprocess_chat(self, chat):
        """Tokenize the chat and get list of sentences to parse.
        This can be swapped out with another preprocessor.
        """
        preprocessed_chat = preprocess.preprocess_chat(chat)
        return preprocessed_chat

    def get_parse(self, chatstr: str) -> Tuple[str, Dict]:
        """This is the function that is called from the perceive() of the agent.
        This function takes in a chat and returns logical form.
        The order is:
        1. Preprocess the incoming chat
        2. check against ground truth and get logical form
        3. query model and get logical form

        Args:
            chatstr (str) : chat or command that needs to be parsed

        Returns:
            Dict: logical form found either in ground truth or from model
        """
        # 1. Preprocess chat
        chat = self.preprocess_chat(chatstr)

        # 2. Get logical form from either ground truth or query the parsing model
        logical_form = self.get_logical_form(chat=chat, parsing_model=self.parsing_model)
        return chat, logical_form

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

    def get_logical_form(self, chat: str, parsing_model) -> Dict:
        # TODO: split this into two steps: gt vs model
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
            time_now = time.time()
        elif self.parsing_model:
            logical_form = parsing_model.query_for_logical_form(chat)
            time_now = time.time()
            logical_form_source = "semantic_parser"
        else:
            logical_form = {"dialogue_type": "NOOP"}
            logging.info("Not found in ground truth, no parsing model initiated. Returning NOOP.")
            time_now = time.time()
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

