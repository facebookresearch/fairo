import abc
import typing

import sophus as sp


class TeleopDeviceReader:
    """Allows for teleoperation using either the keyboard or an Oculus controller"""

    @abc.abstractmethod
    def get_state(self) -> typing.Tuple[bool, sp.SE3, bool]:
        raise NotImplementedError
