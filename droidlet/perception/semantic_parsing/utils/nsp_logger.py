"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import csv


class NSPLogger:
    def __init__(self, filepath, headers):
        """Logger class for the NSP component.

        args:
            filepath (str): Where to log data.
            headers (list): List of string headers to be used in data store.
        """
        self.log_filepath = filepath
        self.init_file_headers(filepath, headers)

    def init_file_headers(self, filepath, headers):
        """Write headers to log file.

        args:
            filepath (str): Where to log data.
            headers (list): List of string headers to be used in data store.
        """
        with open(filepath, "w") as fd:
            csv_writer = csv.writer(fd, delimiter="|")
            csv_writer.writerow(headers)

    def log_dialogue_outputs(self, data):
        """Log dialogue data.

        args:
            filepath (str): Where to log data.
            data (list): List of values to write to file.
        """
        with open(self.log_filepath, "a") as fd:
            csv_writer = csv.writer(fd, delimiter="|")
            csv_writer.writerow(data)
