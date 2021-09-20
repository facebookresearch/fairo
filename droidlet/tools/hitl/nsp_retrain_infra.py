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


class NSPRetrainingJob(DataGenerator):
    def __init__(self):
        super(NSPRetrainingJob, self).__init__()

    def run(self):
        logging.info(f"NSP Retraining Job initialized")

        self.set_finished()
        logging.info(f"NSP Retraining Job finished")


class NSPNewDataListener(JobListener):
    def __init__(self):
        super(NSPNewDataListener, self).__init__()

    def run(self, runner):
        logging.info(f"NSP New Data Listener running")
        while not self.check_is_finished():
            finished = True
            

            for fname in flist:
                with open(fname, "w+") as f:
                    logging.info(f"Change status of file {fname} to finished")
                    f.write("finished")
                finished = False

            if not self.check_parent_finished():
                finished = False

            self.set_finished(finished)


if __name__ == "__main__":
    nsp_rt = NSPRetrainingJob()
    ndl = NSPNewDataListener()
    ndl.add_parent_jobs([nsp_rt])
    runner = TaskRunner()
    runner.register_data_generators([nsp_rt])
    runner.register_job_listeners([ndl])
    runner.run()