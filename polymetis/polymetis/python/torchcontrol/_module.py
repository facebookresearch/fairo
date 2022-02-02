# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Dict, Optional

import torch

log = logging.getLogger(__name__)


class ControlModule(torch.nn.Module):
    """
    Base class for all control blocks.
    Basically a torch.nn.Module with the added functionality of updating parameters
    through a jit-scriptable method.

    Control modules can also contain other control modules, allowing to nest
    them in a tree structure.

    Note: To ensure instance variables with ambiguous types are saved into
    TorchScript, annotate their type as a class variable
    (for example, `_param_dict` below.)
    """

    _param_dict: Dict[str, torch.Tensor]

    def __init__(self):
        self._param_dict = {
            "_": torch.Tensor()
        }  # torchscript can't convert empty dicts
        super().__init__()

    def register_parameter(
        self, name: str, param: Optional[torch.nn.Parameter]
    ) -> None:
        """
        Pre-record all parameters in a dict because Torchscript does not support
        calling 'named_parameters'
        """
        super().register_parameter(name, param)
        self._param_dict[name] = param

    @torch.jit.export
    def update(self, update_dict: Dict[str, torch.Tensor]) -> int:
        """Method for updating module parameters.
        Returns: Error code (0: Successful; 1: Invalid parameter name; 2: Tensor shape mismatch)
        """
        for name in update_dict.keys():
            if name not in self._param_dict.keys():
                return 1
            if self._param_dict[name].shape != update_dict[name].shape:
                return 2
            self._param_dict[name].data.copy_(update_dict[name])

        return 0

    # TODO: Implicitly check input/output format/dimensions?
    # TODO: Warn users against common error of not calling 'super().__init__()'?


class PolicyModule(ControlModule):
    """A policy which can be serialized for use in controller manager.

    Provides a termination flag that controller manager uses to
    stop execution of the policy.

    A PolicyModule is basically a ControlModule with termination
    """

    _terminated: bool
    _log_info: bool

    def __init__(self):
        super().__init__()

        self._terminated = False
        self._log_info = log.getEffectiveLevel() >= logging.INFO

    @torch.jit.export
    def is_terminated(self):
        return self._terminated

    @torch.jit.export
    def set_terminated(self):
        if self._log_info:
            print("Setting Torch policy to terminated.")
        self._terminated = True
