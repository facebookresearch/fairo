"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json

class interactionLogger:
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
        with open (loggingsFile, 'r', encoding='utf-8') as f:
            feeds = json.load(f)
        with open(loggingsFile, 'w', encoding='utf-8') as f:
            # ensure that of interaction_loggings is empty, that it always has []
            feeds.append(data)
            
            # TODO: change this to perform json insertion with characters instead of truncate and repopulate
            f.truncate(0) # clear contents of file. This is pretty poor, should probably move to a database
            json.dump(feeds, f, ensure_ascii=False, indent=4)
