"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import threading

from typing import List

from .data_generator import DataGenerator


class JobListener:
    def __init__(self):
        self._finished = False
        self._started = False
        self._parent_jobs = []

    def run(self):
        pass

    def start(self, *args):
        if not self._started:
            thread = threading.Thread(target=self.run, args=(args))
            thread.daemon = True
            thread.start()
            self._started = True

    def set_finished(self, finished=True):
        self._finished = finished

    def check_is_finished(self) -> bool:
        return self._finished

    def check_parent_finished(self) -> bool:
        finished = True
        for job in self._parent_jobs:
            if not job.check_is_finished():
                finished = False
        return finished

    def add_parent_jobs(self, jobs: List[DataGenerator]):
        self._parent_jobs.extend(jobs)
