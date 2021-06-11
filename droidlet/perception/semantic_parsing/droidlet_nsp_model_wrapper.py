"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import copy
import logging
import pkg_resources
from time import time
from typing import Dict, Tuple
from .utils import preprocess
from .load_and_check_datasets import get_ground_truth
from .nsp_model import DroidletSemanticParsingModel
from droidlet.event import sio
from .utils.nsp_logger import NSPLogger
from .utils.validate_json import JSONValidator


class DroidletNSPModelWrapper(object):
    def __init__(self, opts):
        self.opts = opts

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
        """
        Read the ground truth dataset file: ground_truth/datasets folder
        """
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

    def preprocess_chat(self, chat):
        """Tokenize the chat and get list of sentences to parse.
        This can be swapped out with another preprocessor.
        """
        preprocessed_chat = preprocess.preprocess_chat(chat)
        return preprocessed_chat

    def get_parse(self, chatstr: str) -> Tuple[str, Dict]:
        """This is the function that is called from the tick() of the agent.
        This function takes in a chat and either returns text or logical form.
        The order is:
        1. Preprocess the incoming chat
        2. check against ground truth
        3. query model and get logical form

        Returns:
            str, Dict
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

