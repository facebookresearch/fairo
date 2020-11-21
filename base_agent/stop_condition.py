"""
Copyright (c) Facebook, Inc. and its affiliates.
"""


class StopCondition:
    def __init__(self, agent):
        self.agent = agent

    def check(self) -> bool:
        raise NotImplementedError("Implemented by subclass")


class NeverStopCondition(StopCondition):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "never"

    def check(self):
        return False
