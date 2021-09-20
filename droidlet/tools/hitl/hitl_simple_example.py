"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import os
import re
import time

from typing import List

from utils.data_generator import DataGenerator
from utils.job_listener import JobListener
from utils.task_runner import TaskRunner


WORKDIR = "/Users/ethancarlson/logs/tmp/"

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)

FW_RUN_POLL_TIME = 2
FW_FILE_NUM = 5


class FileWritter(DataGenerator):
    def __init__(self):
        super(FileWritter, self).__init__()
        self.cnt = 0

    def run(self):
        while self.cnt < FW_FILE_NUM:
            time.sleep(FW_RUN_POLL_TIME)
            logging.info(f"Create file #{self.cnt}")
            with open(f"{WORKDIR}{self.cnt}.txt", "w+") as f:
                f.write(f"test#{self.cnt}")
            with open(f"{WORKDIR}{self.cnt}.stat", "w+") as f:
                f.write(f"ready")
            self.cnt += 1

        self.set_finished()
        logging.info(f"File Writter Generator Finished")


class FileListener(JobListener):
    def __init__(self):
        super(FileListener, self).__init__()

    def run(self, runner):
        while not self.check_is_finished():
            finished = True
            fname_re = "\d+.stat"
            flist = []
            for fname in os.listdir(WORKDIR):
                if re.search(fname_re, fname):
                    file_full_path = os.path.join(WORKDIR, fname)
                    with open(file_full_path, "r+") as f:
                        status = f.readline()
                        if status == "ready":
                            logging.info(f"Detect status of file {file_full_path} as ready")
                            flist.append(file_full_path)

            for fname in flist:
                with open(fname, "w+") as f:
                    logging.info(f"Change status of file {fname} to finished")
                    f.write("finished")
                finished = False

            if not self.check_parent_finished():
                finished = False

            self.set_finished(finished)


if __name__ == "__main__":
    fw = FileWritter()
    fl = FileListener()
    fl.add_parent_jobs([fw])
    runner = TaskRunner()
    runner.register_data_generators([fw])
    runner.register_job_listeners([fl])
    runner.run()