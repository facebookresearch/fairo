"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
from unittest.mock import Mock

import craftassist.agent.tasks as tasks
from base_agent.dialogue_objects import BotStackStatus
from dialogue_stack import DialogueStack
from mc_memory import MCAgentMemory


class BotStackStatusTest(unittest.TestCase):
    def setUp(self):
        self.agent = Mock(["send_chat"])
        self.memory = MCAgentMemory()
        self.agent.memory = self.memory
        self.dialogue_stack = DialogueStack(self.agent, self.memory)
        self.dialogue_stack.append(
            BotStackStatus(
                agent=self.agent, memory=self.memory, dialogue_stack=self.dialogue_stack
            )
        )

    def test_move(self):
        self.memory.task_stack_push(tasks.Move(self.agent, {"target": (42, 42, 42)}))
        self.memory.add_chat("test_agent", "test chat: where are you going?")
        self.dialogue_stack.step()
        self.agent.send_chat.assert_called()


if __name__ == "__main__":
    unittest.main()
