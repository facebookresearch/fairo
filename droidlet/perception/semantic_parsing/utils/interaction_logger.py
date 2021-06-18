"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json

class interactionLogger:
    def __init__(self, filepath, headers):
        """Logger class for the NSP component.

        args:
            filepath (str): Where to log data.
            headers (list): List of string headers to be used in data store.
        """
        self.log_filepath = filepath
        self.init_file_headers(filepath, headers)

    def logInteraction(self, data):
        """Log interaction data.

        args:
            filepath (str): Where to log data.
            data (list): List of values to write to file.
        """
      loggingsFile = 'loggings.json'

      with open(loggingsFile) as f:
          data = json.load(f)

      data.append(data)

      with open(loggingsFile, 'w') as f:
        json.dump(data, f)
