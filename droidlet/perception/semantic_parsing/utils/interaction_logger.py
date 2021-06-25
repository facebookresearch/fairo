"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json
import os.path

class InteractionLogger:
    def __init__(self, filepath):
        """Logger class for the NSP component.

        args:
            filepath (str): Where to log data.
            headers (list): List of string headers to be used in data store.
        """
        self.log_filepath = filepath
        self.createdFile = False


    def create_loggings_file(self, filepath):
        self.createdFile = True

        if not os.path.isfile(self.log_filepath):
            with open(self.log_filepath, 'w') as f:
                json.dump([], f, ensure_ascii=False, indent=4)

    def logInteraction(self, data):

        """Log interaction data.

        args:
            filepath (str): Where to log data.
            data (list): List of values to write to file.
        """
        
        if not self.createdFile:
            self.create_loggings_file(self.log_filepath)
        
        loggingsFile =  self.log_filepath

        with open (loggingsFile, 'r', encoding='utf-8') as f:
            # read json file
            feeds = json.load(f)
        with open(loggingsFile, 'w', encoding='utf-8') as f:
            # ensure that of interaction_loggings is empty, that it always has []
            feeds.append(data)
            # replace the original json file with what is in the updated variable
            json.dump(feeds, f, ensure_ascii=False, indent=4)
