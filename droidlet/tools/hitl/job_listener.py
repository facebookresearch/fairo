"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import threading
import time

from abc import ABC, abstractmethod
from typing import List

from .data_generator import DataGenerator


class JobListener:
    """
    This class is an abstraction of a job listener which listen to a certain location,
    waits for an event/data to occur and perform some actions. Note that the actions 
    should be high level operations and should be differentiated from DataGenerator.
    You should always encapsulate the logic into a DataGenerator job and push the job to runner
    if you want to process the listenered data and generate new kinds of data (or none),
    instead of encompassing the logic into the listener run() method.

    It maintains details such as start time, timeout value, etc. and is responsible for
    starting the listener process, monitoring the status, ending the listener at the correct
    time and cleaning up the process properly when terminated.    
    """

    def __init__(self, timeout: float = -1) -> None:
        self._finished = False
        self._started = False
        self._parent_jobs = []
        self.start_time = time.time()
        self.timeout = (
            timeout
        )  # in minutes, -1 if no timeout is set (run indefinitely until killed by runner)

    @abstractmethod
    def run(self) -> None:
        """
        This method must be implemented by subclass.
        The logic of how the data is generated should be put here.
        """
        raise NotImplementedError()

    def start(self, *args) -> None:
        """
        Start a separate thread which call run() fn and keep running util terminated
        """
        if not self._started:
            thread = threading.Thread(target=self.run, args=(args))
            thread.daemon = True
            thread.start()
            self._started = True

    def set_finished(self, finished: bool = True) -> None:
        """
        Set task finish status
        """
        self._finished = finished

    def check_is_finished(self) -> bool:
        """
        Return if the task has finished
        """
        return self._finished

    def check_parent_finished(self) -> bool:
        """
        Return if all parent jobs have finished

        This should be used together with timeout to determine whether the listener should
        be terminated. For example, you may want to terminate the listener process when all
        parent DataGenerator jobs have finished (stop to produce more data), even timeout is
        set to -1 (run indefinitely)
        """
        finished = True
        for job in self._parent_jobs:
            if not job.check_is_finished():
                finished = False
        return finished

    def add_parent_jobs(self, jobs: List[DataGenerator]) -> None:
        """
        Register new parent jobs.
        """
        self._parent_jobs.extend(jobs)

    def check_is_timeout(self) -> bool:
        """
        Return if the task has timed out
        """
        if self.timeout == -1:
            return False

        if time.time() - self.start_time < self.timeout * 60:
            return False

        return True

    def get_remaining_time(self) -> float:
        """
        Return remaining time before task timeout
        """
        return self.timeout - ((time.time() - self.start_time) // 60)
