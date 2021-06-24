"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json

class InteractionLogger:
    def __init__(self, filepath):
        """Logger class for the NSP component.

        args:
            filepath (str): Where to log data.
            headers (list): List of string headers to be used in data store.
        """
        self.log_filepath = filepath

    def logInteraction(self, data):

        """Log interaction data.

        args:
            filepath (str): Where to log data.
            data (list): List of values to write to file.
        """

        loggingsFile =  self.log_filepath

        # TODO: Instead of reading the json file and storing it in a variable, then updating the variable, write directly into the file
        with open (loggingsFile, 'r', encoding='utf-8') as f:
            # read json file
            feeds = json.load(f)
        with open(loggingsFile, 'w', encoding='utf-8') as f:
            # ensure that of interaction_loggings is empty, that it always has []
            feeds.append(data)
            # replace the original json file with what is in the updated variable
            json.dump(feeds, f, ensure_ascii=False, indent=4)
