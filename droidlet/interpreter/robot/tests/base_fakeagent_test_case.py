"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from unittest.mock import Mock
import copy

from .fake_agent import FakeAgent
from .world import World, Opt, SimpleHuman, make_human_opts


class BaseFakeAgentTestCase(unittest.TestCase):
    def setUp(self, agent_opts=None):
        spec = {"players": [SimpleHuman(make_human_opts())], "agent": {"pos": (0, 0)}}
        world_opts = Opt()
        world_opts.sl = 32
        self.world = World(world_opts, spec)
        self.agent = FakeAgent(self.world, opts=agent_opts)

        # More helpful error message to encourage test writers to use self.set_looking_at()
        self.agent.get_player_line_of_sight = Mock(
            side_effect=NotImplementedError(
                "Cannot call into C++ function in this unit test. "
                + "Call self.set_looking_at() to set the return value"
            )
        )

        self.speaker = self.agent.get_other_players()[0].name
        self.agent.perceive()



    def handle_logical_form(
        self, d, chatstr: str = "", answer: str = None, stop_on_chat=False, max_steps=10000
    ):
        """Handle a logical form and call self.flush()

        If "answer" is specified and a question is asked by the agent, respond
        with this string.

        If "stop_on_chat" is specified, stop iterating if the agent says anything
        """
        chatstr = chatstr or "TEST {}".format(d)
        self.add_incoming_chat(chatstr, self.speaker)
        self.agent.set_logical_form(d, chatstr, self.speaker)
        changes = self.flush(max_steps, stop_on_chat=stop_on_chat)
        if len(self.agent.dialogue_manager.dialogue_stack) != 0 and answer is not None:
            self.add_incoming_chat(answer, self.speaker)
            new_changes = self.flush(max_steps, stop_on_chat=stop_on_chat)
            changes[1] = new_changes[1]
        return changes

    def flush(self, max_steps=10000, stop_on_chat=False):
        """Run the agant's step until task and dialogue stacks are empty.

        If "stop_on_chat" is specified, stop iterating if the agent says anything

        Return the set of blocks that were changed.
        """
        if stop_on_chat:
            self.agent.clear_outgoing_chats()

        world_before = copy.deepcopy(self.agent.world.get_info())

        for _ in range(max_steps):
            self.agent.step()
            if self.agent_should_stop(stop_on_chat):
                break

        # get changes
        world_after = copy.deepcopy(self.world.get_info())
        changes = [world_before, world_after]
        return changes

    def agent_should_stop(self, stop_on_chat=False):
        stop = False
        if (
            len(self.agent.dialogue_manager.dialogue_stack) == 0
            and not self.agent.memory.task_stack_peek()
        ):
            stop = True
        # stuck waiting for answer?
        _, task_mems = self.agent.memory.basic_search(
            "SELECT MEMORY FROM Task WHERE (action_name=awaitresponse AND prio>-1)"
        )
        if task_mems and not any([m.finished for m in task_mems]):
            stop = True
        if stop_on_chat and self.agent.get_last_outgoing_chat():
            stop = True
        return stop

    def set_looking_at(self, xyz):
        """Set the return value for C++ call to get_player_line_of_sight; xyz
        is a tuple of floats."""

        self.agent.get_player_line_of_sight = Mock(return_value=xyz)

    #    def add_object(self, xyzbms: List[Block], origin: XYZ = (0, 0, 0)) -> VoxelObjectNode:
    #        return self.agent.add_object(xyzbms, origin)

    def add_incoming_chat(self, chat: str, speaker_name: str):
        """Add a chat to memory as if it was just spoken by SPEAKER."""
        self.world.chat_log.append("<" + speaker_name + ">" + " " + chat)

    #        self.agent.memory.add_chat(self.agent.memory.get_player_by_name(self.speaker).memid, chat)

    def last_outgoing_chat(self) -> str:
        return self.agent.get_last_outgoing_chat()

    def get_speaker_pos(self):
        return self.agent.memory.get_player_by_name(self.speaker).pos
