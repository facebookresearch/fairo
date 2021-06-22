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
        print ("LOGGING AN INTERACTION")
        print ("LOGGING AN INTERACTION")
        print ("LOGGING AN INTERACTION")
        print ("LOGGING AN INTERACTION")
        print ("LOGGING AN INTERACTION")

        """Log interaction data.

        args:
            filepath (str): Where to log data.
            data (list): List of values to write to file.
        """
        
        print (self.log_filepath)
        loggingsFile =  self.log_filepath

        # with open(loggingsFile) as f:
        #     print ("f is")
        #     print (f)

        #     file = {}
        #     if (f):
        #         file = json.load(f)

        # file.append(data)

        with open(loggingsFile, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # with open(loggingsFile, 'w') as f:
        #     json.dump(file, f)
