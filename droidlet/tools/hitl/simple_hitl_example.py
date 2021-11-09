"""
Copyright (c) Facebook, Inc. and its affiliates.

This is a simple example demonstrating the usage of HiTL library

It registers:
1. a FileWriter DataGenerator which will generate certain kind of file to your local FS
2. a FileListener JobListener which will listen to your local FS for files FileWriter generated
   and perform in-place transformation of them -- overwriting the content of them.
"""

import logging
import os
import re
import time

from typing import List

from .data_generator import DataGenerator
from .job_listener import JobListener
from .task_runner import TaskRunner


WORKDIR = "/private/home/yuxuans/Workspace/tmp/droidlet/droidlet/tools/hitl/tmp/"

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


class FileWriter(DataGenerator):
    """
    This is a simple example demonstrating the purpose of a DataGenerator.

    A DataGenrator should:
    - Take in some input data of all kinds of format (can be raw text/image data, requests, etc.)
    - Generate some output data of all kinds of format (can be transformation of input data, new data kind, even in-place change, etc.)

    For example, this FileWriter class will:
    - Takes in a request, asking for 5 new files
    - Generate 5 new files all with extension name .stat. The file contents are all the same: a single-line text 'ready'

    """
    def __init__(self, file_num: int) -> None:
        super(FileWriter, self).__init__()
        self.file_num = file_num
        self.cnt = 0

    def run(self) -> None:
        while self.cnt < self.file_num:
            time.sleep(FW_RUN_POLL_TIME)
            logging.info(f"Create file #{self.cnt}")
            with open(f"{WORKDIR}{self.cnt}.stat", "w+") as f:
                f.write(f"ready")
            self.cnt += 1

        self.set_finished()
        logging.info(f"File Writer Generator Finished")


class FileListener(JobListener):
     """
    This is a simple example demonstrating the purpose of a JobListener.

    A DataGenrator should:
    - Listen to a certain location for specific data
    - Assign resources and perform certain actions

    For example, this FileListener class will:
    - Keep checking the given local path for files with extension .stat whose content is a single-line text 'ready'
    - Perform in-place transformation of the file -- overwritting the file content to be a single-line text 'finished'

    """

    def __init__(self) -> None:
        super(FileListener, self).__init__()

    def run(self, runner: TaskRunner) -> None:
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
    fw = FileWriter(file_num=FW_FILE_NUM)
    fl = FileListener()
    fl.add_parent_jobs([fw])
    runner = TaskRunner()
    runner.register_data_generators([fw])
    runner.register_job_listeners([fl])
    runner.run()
