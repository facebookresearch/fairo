from abc import ABC, abstractmethod


class PoseInitializer(ABC):
    @abstractmethod
    def get_pose(self):
        return NotImplemented
