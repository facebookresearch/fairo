"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging

from base_agent.base_util import NextDialogueStep, ErrorWithResponse


class DialogueStack(object):
    """This class organizes and steps DialogueObjects."""

    def __init__(self, agent, memory):
        self.agent = agent
        self.memory = memory
        self.stack = []

    def __getitem__(self, i):
        """Get the ith item on the stack """
        return self.stack[i]

    def peek(self):
        """Get the item on top of the DialogueStack"""
        if self.stack:
            return self.stack[-1]
        else:
            return None

    def clear(self):
        """clear current stack"""
        self.old_stack = self.stack
        self.stack = []

    def append(self, dialogue_object):
        """Append a dialogue_object to stack"""
        self.stack.append(dialogue_object)

    def append_new(self, cls, *args, **kwargs):
        """Construct a new DialogueObject and append to stack"""
        self.stack.append(
            cls(agent=self.agent, memory=self.memory, dialogue_stack=self, *args, **kwargs)
        )

    def step(self):
        """Process and step through the top-of-stack dialogue object."""
        if len(self.stack) > 0:
            # WARNING: check_finished increments the DialogueObject's current_step counter
            while len(self.stack) > 0 and self.stack[-1].check_finished():
                del self.stack[-1]

            if len(self.stack) == 0:
                return

            try:
                output_chat, step_data = self.stack[-1].step()
                if output_chat:
                    self.agent.send_chat(output_chat)

                # Update progeny_data of the current DialogueObject
                if len(self.stack) > 1 and step_data is not None:
                    logging.debug("Update progeny_data={} stack={}".format(step_data, self.stack))
                    self.stack[-2].update_progeny_data(step_data)

            except NextDialogueStep:
                return
            except ErrorWithResponse as err:
                self.stack[-1].finished = True
                self.agent.send_chat(err.chat)
                return

    def __len__(self):
        """Length of stack"""
        return len(self.stack)
