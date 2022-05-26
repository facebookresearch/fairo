from abc import ABC, abstractmethod


class LidarABC(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_latest_scan(self):
        pass
