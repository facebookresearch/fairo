"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import threading
import time

from abc import ABC, abstractmethod


class DataGenerator(ABC):
    """
    This class is an abstraction of a data generator which generates all kinds of data
    and stores them at give location (local, cloud, etc.)

    It maintains details such as start time, timeout value, etc. and is responsible for
    starting the task process, monitoring the status, ending the task at the correct
    time and cleaning up the process properly when terminated.
    """

    def __init__(self, timeout: float = -1) -> None:
        self._finished = False
        self._started = False
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
            self.start_time = time.time()
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

    def shutdown(self) -> None:
        pass
