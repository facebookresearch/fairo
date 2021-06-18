"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json


class jsonLogger:
    def __init__(self, filepath):
        """Logger class for the NSP component.

        args:
            filepath (str): Where to log data.
        """
        self.log_filepath = filepath

    def log_interaction(self, logDict:dict):
        """Log dialogue data.

        args:
            logDict (dict): List of values to write to file.
        """
        with open(self.log_filepath) as f:
            data = json.load(f)

        data.append(logDict)

        with open(loggingsFile, 'w') as f:
            json.dump(data, f)
