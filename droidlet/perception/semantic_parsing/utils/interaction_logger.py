"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json
import os.path

class InteractionLogger:
    def __init__(self):
        """Logger class for the NSP component.

        args:
            filepath (str): Where to log data.
            headers (list): List of string headers to be used in data store.
        """

    def logInteraction(self, data):

        """Log interaction data.

        args:
            filepath (str): Where to log data.
            data (list): List of values to write to file.
        """
        
        if (data["session_id"]):
            # get session_id
            session_id = data["session_id"]
            filePath = "interaction_loggings_" + session_id + ".json"

            # if file path does not exist, write empty json into the file
            if not os.path.isfile(filePath):
                with open(filePath, 'w') as f:
                    json.dump([], f, ensure_ascii=False, indent=4)
            
            with open(filePath, 'r', encoding='utf-8') as f:
                # read json file
                feeds = json.load(f)
            with open(filePath, 'w', encoding='utf-8') as f:
                # ensure that of interaction_loggings is empty, that it always has []
                feeds.append(data)
                # replace the original json file with what is in the updated variable
                json.dump(feeds, f, ensure_ascii=False, indent=4)
