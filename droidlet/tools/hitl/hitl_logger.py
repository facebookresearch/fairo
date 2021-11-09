"""
Copyright (c) Facebook, Inc. and its affiliates.

Part of it is taken from: https://stackoverflow.com/questions/15870380/python-custom-logging-across-all-modules
"""

import logging
import os
import sys

LOG_ROOT_PATH = ""

def set_logging_root_path(root_path: str) -> None:
    LOG_ROOT_PATH = root_path

def get_logger(name: str = 'root', loglevel: str = 'INFO'):
  logger = logging.getLogger(name)

  if logger.handlers:
    return logger

  else:
    loglevel = getattr(logging, loglevel.upper(), logging.INFO)
    logger.setLevel(loglevel)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    log_file_path = os.path.join(LOG_ROOT_PATH, name)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    return logger