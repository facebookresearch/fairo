"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import os
from typing import Dict
from droidlet.dialog.ttad.ttad_transformer_model.query_model import TTADBertModel as Model


class DroidletSemanticParsingModel:
    def __init__(self, models_dir, data_dir):
        """The SemanticParsingModel converts natural language
        commands to logical forms.

        Instantiates the ML model used for semantic parsing, ground truth data
        directory and sets up the NSP logger to save dialogue outputs.

        NSP logger schema:
        - command (str): chat command received by agent
        - action_dict (dict): logical form output
        - source (str): the source of the logical form, eg.
        model or ground truth
        - agent (str): the agent that processed the command
        - time (int): current time in UTC

        args:
            models_dir (str): path to semantic parsing models
            data_dir (str): path to ground truth data directory
        """
        # Instantiate the main model
        ttad_model_dir = os.path.join(models_dir, "ttad_bert_updated")
        logging.info("using model_dir={}".format(ttad_model_dir))

        if os.path.isdir(data_dir) and os.path.isdir(ttad_model_dir):
            self.model = Model(model_dir=ttad_model_dir, data_dir=data_dir)
        else:
            raise NotADirectoryError

    def query_for_logical_form(self, chat: str) -> Dict:
        """Get logical form output for a given chat command.
        First check the ground truth file for the chat string. If not
        in ground truth, query semantic parsing model to get the output.

        Args:
            chat (str): Input chat provided by the user.

        Return:
            Dict: Logical form representation of the task. See paper for more
                in depth explanation of logical forms:
                https://arxiv.org/abs/1907.08584

        Examples:
            >>> query_for_logical_form("destroy this", model)
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
        logging.info("Querying the semantic parsing model")
        logical_form = self.model.parse(chat=chat)
        return logical_form
