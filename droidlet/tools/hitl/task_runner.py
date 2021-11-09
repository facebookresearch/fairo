"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import time

from typing import List

from .data_generator import DataGenerator
from .job_listener import JobListener

RUN_POLL_TIME = 5

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)


class TaskRunner:
    """
    This class acts as a job scheduler that is responsible for registering data generators &
    job listeners and scheduling the runnings of registered tasks.
    """

    def __init__(self) -> None:
        self._data_generators = []
        self._job_listeners = []
        self._finished = False

    def register_data_generators(self, data_generators: List[DataGenerator]) -> None:
        """
        Register new data generator tasks
        """
        for data_generator in data_generators:
            self._data_generators.append(data_generator)

    def register_job_listeners(self, job_listeners: List[JobListener]) -> None:
        """
        Register new job listener tasks
        """
        for job_listener in job_listeners:
            self._job_listeners.append(job_listener)

    def run(self) -> None:
        """
        Assign resources and schedule task runnings
        For now it just start all the registered tasks
        """
        try:
            while not self._finished:
                for data_generator in self._data_generators:
                    data_generator.start()

                for job_listener in self._job_listeners:
                    job_listener.start(self)

                self._finished = self._check_is_finished()
                logging.info(f"Task is running...")
                time.sleep(RUN_POLL_TIME)
        except (KeyboardInterrupt, Exception) as e:
             logging.error(f"Encountered errors during task running, shutting down everything")
             self._clean_up()


    def _clean_up(self) -> None:
        for data_generator in self._data_generators:
            data_generator.shutdown()

        for job_listener in self._job_listeners:
            job_listener.shutdown()


    def _check_is_finished(self) -> bool:
        """
        Return if the runner has finished.

        The runner is considered finished when all registered tasks have finsihed
        """
        finished = True
        if any(
            [
                data_generator.check_is_finished() == False
                for data_generator in self._data_generators
            ]
        ):
            logging.debug("Remaining data generator jobs, not finished...")
            finished = False
        if any(
            [job_listener.check_is_finished() == False for job_listener in self._job_listeners]
        ):
            logging.debug("Remaining job listener jobs, not finished...")
            finished = False
        return finished
