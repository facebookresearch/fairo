"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import random

from string_lists import MAP_YES, MAP_NO
from base_util import pos_to_np
from enum import Enum


class DialogueObject(object):
    """DialogueObject class controls the agent's use of the dialogue stack.

    Args:
        agent: the agent process
        memory: agent's memory
        dialogue_stack : A stack on which dialogue objects are placed
        finished: whether this object has finished processing
        awaiting_response: whether this object is awaiting the speakers response to a question
        max_steps: finish after this many steps to avoid getting stuck
        current_step: current step count
        progeny_data: data from progeny DialogueObjects, for example used to answer a clarification

    Usage:

.. code-block:: python

        class DummyGetMemoryHandler(DialogueObject):
            def __init__(self, speaker_name: str, action_dict: Dict, **kwargs):
                # initialize everything
                super().__init__(**kwargs)
                self.speaker_name = speaker_name
                self.action_dict = action_dict

            def step(self) -> Tuple[Optional[str], Any]:
                # check for dialogue type "GET_MEMORY"
                assert self.action_dict["dialogue_type"] == "GET_MEMORY"
                memory_type = self.action_dict["filters"]["memory_type"]
                if memory_type == "AGENT" or memory_type == "REFERENCE_OBJECT":
                    return self.handle_reference_object() # handle these two by writing them to memory
                else:
                    raise ValueError("Unknown memory_type={}".format(memory_type))
                # mark as finished
                self.finished = True

    """

    def __init__(self, agent, memory, dialogue_stack, max_steps=50):
        self.agent = agent
        self.memory = memory
        self.dialogue_stack = dialogue_stack
        self.finished = False
        self.awaiting_response = False
        self.max_steps = max_steps
        self.current_step = 0
        self.progeny_data = (
            []
        )  # this should have more structure and some methods for adding/accessing?

    def step(self):
        """the Dialogue Stack runs this objects .step();

        Returns:
            chat: string to be uttered by the agent (or None)
            progeny_data: data for the parent of this object (or None)
        """
        raise NotImplementedError("Subclasses must implement step()")

    def update_progeny_data(self, data):
        self.progeny_data.append(data)

    def check_finished(self):
        """Check if the object is finished processing."""
        self.current_step += 1
        if self.current_step == self.max_steps:
            logging.error("Stepped {} {} times, finishing".format(self, self.max_steps))
            self.finished = True
        return self.finished

    def __repr__(self):
        return str(type(self))


class AwaitResponse(DialogueObject):
    """This class represents a sub-type of DialogueObject to await
    a response from the user.

    Args:
        init_time: initial time
        response: the response to the question asked
        wait_time: how long should we await the response
        awaiting_response: a flag to mark where we are awaiting the response
    """

    def __init__(self, wait_time=800, **kwargs):
        super().__init__(**kwargs)
        self.init_time = self.memory.get_time()
        self.response = []
        self.wait_time = wait_time
        self.awaiting_response = True

    def step(self):
        """Wait for wait_time for an answer. Mark finished when a chat comes in."""
        chatmem = self.memory.get_most_recent_incoming_chat(after=self.init_time + 1)
        if chatmem is not None:
            self.finished = True
            return "", {"response": chatmem}
        if self.memory.get_time() - self.init_time > self.wait_time:
            self.finished = True
            # FIXME this shouldn't return data
            return "Okay! I'll stop waiting for you to answer that.", {"response": None}
        return "", None


class Say(DialogueObject):
    """This class represents a sub-type of DialogueObject to say / send a chat
    to the user.

    Args:
        response_options: a list of responses to pick the final response from

    """

    def __init__(self, response_options, **kwargs):
        super().__init__(**kwargs)
        if len(response_options) == 0:
            raise ValueError("Cannot init a Say with no response options")

        if type(response_options) is str:
            self.response_options = [response_options]
        else:
            self.response_options = response_options

    def step(self):
        """Return one of the response_options."""
        self.finished = True
        return random.choice(self.response_options), None


class BotCapabilities(Say):
    """This class represents a sub-type of the Say DialogueObject above to answer
    something about the current capabilities of the bot, to the user.

    """

    def __init__(self, **kwargs):
        response_options = [
            'Try looking at something and tell me "go there"',
            'Try looking at a structure and tell me "destroy that"',
            'Try looking somewhere and tell me "build a wall there"',
            "Try building something and giving it a name",
            "Try naming something and telling me to build it",
        ]
        super().__init__(response_options, **kwargs)


class GreetingType(Enum):
    """Types of bot greetings."""

    HELLO = "hello"
    GOODBYE = "goodbye"


class BotGreet(Say):
    """This class represents a sub-type of the Say DialogueObject above to greet
    the user as a reply to a greeting.

    """

    def __init__(self, greeting_type, **kwargs):
        if greeting_type == GreetingType.GOODBYE.value:
            response_options = ["goodbye", "bye", "see you next time!"]
        else:
            response_options = ["hi there!", "hello", "hey", "hi"]
        super().__init__(response_options, **kwargs)


class BotStackStatus(DialogueObject):
    """This class represents a sub-type of the DialogueObject to answer
    questions about the current status of the bot, to the user.

    Args:
        ing_mapping: A map from task name to present continuous tense language.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ing_mapping = {
            "Build": "building",
            "Destroy": "excavating",
            "Dig": "digging",
            "Move": "moving",
        }

    def step(self):
        """return the current task name status."""
        self.finished = True

        task_mem = self.memory.task_stack_find_lowest_instance(list(self.ing_mapping.keys()))
        if task_mem is None:
            answer_options = [
                "Idle. You can tell me what to do!",
                "I am doing nothing.",
                "Nothing. Waiting for your command.",
            ]
        else:
            doing = self.ing_mapping[task_mem.task.__class__.__name__]
            answer_options = ["I am " + doing, doing]
        return random.choice(answer_options), None


class GetReward(DialogueObject):
    """This class represents a sub-type of the DialogueObject to register feedback /
    reward given by the user in the form of chat.

    """

    def step(self):
        """associate pos / neg reward to chat memory."""
        self.finished = True
        chatmem = self.memory.get_most_recent_incoming_chat()
        if chatmem.chat_text in [
            "that is wrong",
            "that was wrong",
            "that was completely wrong",
            "not that",
            "that looks horrible",
            "that is not what i asked",
            "that is not what i told you to do",
            "that is not what i asked for" "not what i told you to do",
            "you failed",
            "failure",
            "fail",
            "not what i asked for",
        ]:
            self.memory.tag(chatmem.memid, "neg_reward")
            # should probably tag recent actions too? not just the text of the chat
        elif chatmem.chat_text in [
            "good job",
            "that is really cool",
            "that is awesome",
            "awesome",
            "that is amazing",
            "that looks good",
            "you did well",
            "great",
            "good",
            "nice",
        ]:
            self.memory.tag(chatmem.memid, "pos_reward")
        return "Thanks for letting me know.", None


class BotLocationStatus(DialogueObject):
    """This class represents a sub-type of the DialogueObject to answer questions
    about the current location of the bot.

    """

    def step(self):
        """Extract bot's current location."""
        self.finished = True
        # Get the agent's current location
        agent_pos = pos_to_np(self.agent.get_player().pos)
        agent_coordinates = " , ".join([str(pos) for pos in agent_pos])
        answer_options = [
            "I am here at location : %r" % (agent_coordinates),
            "I am standing at : %r" % (agent_coordinates),
            "I am at : %r" % (agent_coordinates),
        ]
        return random.choice(answer_options), None


class BotMoveStatus(DialogueObject):
    """This class represents a sub-type of the DialogueObject to answer questions
    about where the bot is moving to.

    """

    def step(self):
        """Extract bot's move target coordinates."""
        self.finished = True
        task = self.memory.task_stack_find_lowest_instance("Move")
        if task is None:
            answer_options = [
                "I am not going anywhere",
                "I am not heading anywhere",
                "I am not off to anywhere",
            ]
        else:
            target_coordinates = " , ".join([str(pos) for pos in task.target])
            answer_options = [
                "I am heading to location : %r" % (target_coordinates),
                "I am walking over to : %r" % (target_coordinates),
                "I am moving to : %r" % (target_coordinates),
            ]
        return random.choice(answer_options), None


class ConfirmTask(DialogueObject):
    """This class represents a sub-type of the DialogueObject to ask a clarification
    question about something.

    Args:
        question: the question to ask the user
        tasks: list of task objects
        asked: flag to denote whether the clarification has been asked for

    """

    def __init__(self, question, tasks, **kwargs):
        super().__init__(**kwargs)
        self.question = question  # chat text that will be sent to user
        self.tasks = tasks  # list of Task objects, will be pushed in order
        self.asked = False

    def step(self):
        """Ask a confirmation question and wait for response."""
        # Step 1: ask the question
        if not self.asked:
            self.dialogue_stack.append_new(AwaitResponse)
            self.dialogue_stack.append_new(Say, self.question)
            self.asked = True
            return "", None

        # Step 2: check the response and add the task if necessary
        self.finished = True
        if len(self.progeny_data) == 0:
            return None, None
        if hasattr(self.progeny_data[-1]["response"], "chat_text"):
            response_str = self.progeny_data[-1]["response"].chat_text
        else:
            response_str = "UNK"
        if response_str in MAP_YES:
            for task in self.tasks:
                self.memory.task_stack_push(task)
        return None, None


class ConfirmReferenceObject(DialogueObject):
    """This class represents a sub-type of the DialogueObject to confirm if the
    reference object is correct.

    Args:
        bounds: general area of reference object to point at
        pointed: flag determining whether the agent pointed at the area
        asked: flag determining whether the confirmation was asked for

    """

    def __init__(self, reference_object, **kwargs):
        super().__init__(**kwargs)
        r = reference_object
        if hasattr(r, "get_point_at_target"):
            self.bounds = r.get_point_at_target()
        else:
            # this should be an error
            self.bounds = tuple(np.min(r, axis=0)) + tuple(np.max(r, axis=0))
        self.pointed = False
        self.asked = False

    def step(self):
        """Confirm the block object by pointing and wait for answer."""
        if not self.asked:
            self.dialogue_stack.append_new(Say, "do you mean this?")
            self.asked = True
            return "", None
        if not self.pointed:
            self.agent.point_at(self.bounds)
            self.dialogue_stack.append_new(AwaitResponse)
            self.pointed = True
            return "", None
        self.finished = True
        if len(self.progeny_data) == 0:
            output_data = None
        else:
            if hasattr(self.progeny_data[-1]["response"], "chat_text"):
                response_str = self.progeny_data[-1]["response"].chat_text
            else:
                response_str = "UNK"
            if response_str in MAP_YES:
                output_data = {"response": "yes"}
            elif response_str in MAP_NO:
                output_data = {"response": "no"}
            else:
                output_data = {"response": "unkown"}
        return "", output_data
