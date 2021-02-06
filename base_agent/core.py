"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os

class BaseAgent:
    def __init__(self, opts, name=None):
        self.opts = opts
        self.name = name or "bot"
        self.count = 0
        self.init_memory()
        self.init_controller()
        self.init_perception()

    def start(self):
        self.bootstrap_agent()
        while True:  # count forever
            try:
                self.step()
            except Exception as e:
                self.handle_exception(e)

    def step(self):
        self.perceive()
        self.memory.update(self)
        # maybe place tasks on the stack, based on memory/perception
        self.controller_step()
        # step topmost task on stack
        self.task_step()
        self.count += 1

    def perceive(self):
        """
        Get information from the world and store it in memory.
        """
        raise NotImplementedError

    def controller_step(self):
        """
        interpret commands, carry out dialogues, etc.  place tasks on the task stack.
        """
        raise NotImplementedError

    def task_step(self):
        """
        run the current task on the stack. interact with the world
        """
        raise NotImplementedError

    def handle_exception(self, e):
        """
        handle/log exceptions
        """
        raise NotImplementedError

    def init_event_handlers(self):
        """
        initialize any socket server handlers here
        """
        raise NotImplementedError
