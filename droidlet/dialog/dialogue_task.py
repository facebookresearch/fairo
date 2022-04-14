"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import random
import json

from droidlet.dialog.string_lists import MAP_YES, MAP_NO
from enum import Enum
from droidlet.task.task import Task, maybe_task_list_to_control_block
from droidlet.memory.memory_nodes import TaskNode


class AwaitResponse(Task):
    """This Task awaits a response from the user.  it is blocking

    Args:
        agent: the droidlet agent that runs the task, needs a .send_chat() method
        init_time: initial time
        wait_time: how long should we await the response
        awaiting_response: a flag to mark where we are awaiting the response
    """

    def __init__(self, agent, task_data={}):
        task_data["blocking"] = True
        super().__init__(agent, task_data=task_data)
        self.init_time = self.agent.memory.get_time()
        self.wait_time = task_data.get("wait_time", 5000)
        self.awaiting_response = True
        self.asker_memid = task_data.get("asker_memid")
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        """Wait for wait_time for an answer. Mark finished when a chat comes in."""
        # FIXME: don't need a memory method for this, use query
        chatmem = self.agent.memory.get_most_recent_incoming_chat(after=self.init_time + 1)
        if chatmem is not None:
            self.finished = True
            if self.asker_memid:
                self.agent.memory.add_triple(
                    subj=self.asker_memid, pred_text="dialogue_task_response", obj=chatmem.memid
                )
            return
        if self.agent.memory.get_time() - self.init_time > self.wait_time:
            self.finished = True
            self.agent.send_chat("Okay! I'll stop waiting for you to answer that.")
        return


class Say(Task):
    """This Task to says / sends a chat
    to the user.

    Args:
        response_options: a list of responses to pick the final response from

    """

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data={})
        response_options = task_data.get("response_options")
        if not response_options:
            raise ValueError("Cannot init a Say with no response options")

        if type(response_options) is str:
            self.response_options = [response_options]
        else:
            self.response_options = response_options
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        """Return one of the response_options."""
        self.finished = True
        self.agent.send_chat(random.choice(self.response_options))


class BotCapabilities(Say):
    """This class represents a sub-type of the Say Task to answer
    something about the current capabilities of the bot, to the user.

    """

    def __init__(self, agent, task_data={}):
        # Should put these in a config file somewhere...
        response_options = [
            'Try looking at something and tell me "go there"',
            "Try asking to get something for you",
            "Try asking me to dance",
            "Try asking me to point at something",
            "Try asking me to drop whatever is in my hand",
        ]
        super().__init__(agent, {"response_options": response_options})


class ConfirmTask(Task):
    """Confirm that a Task should be executed

    Args:
        question: the question to ask the user
        tasks: list of task objects
        asked: flag to denote whether the clarification has been asked for

    """

    def __init__(self, agent, task_data={}):
        task_data["blocking"] = True
        super().__init__(agent, task_data=task_data)
        self.question = task_data.get("question")  # chat text that will be sent to user
        self.task_memids = task_data.get(
            "task_memids"
        )  # list of Task objects, will be pushed in order
        self.asked = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        """Ask a confirmation question and wait for response."""
        # Step 1: ask the question
        if not self.asked:
            task_list = [
                Say(self.agent, {"response_options": self.question}),
                AwaitResponse(self.agent, {"asker_memid": self.memid}),
            ]
            self.add_child_task(maybe_task_list_to_control_block(task_list, self.agent))
            self.asked = True
            return
        # Step 2: check the response and add the task if necessary
        self.finished = True
        # FIXME: change this to sqly when syntax for obj searches is settled:
        # search for a response to the confirmation question, which will be a triple
        # (self.memid, "dialogue_task_reponse", chat_memid)
        t = self.agent.memory.get_triples(subj=self.memid, pred_text="dialogue_task_response")
        if not t:
            return
        chat_mems = [self.agent.memory.get_mem_by_id(triples[2]) for triples in t]
        if any([c.chat_text in MAP_YES for c in chat_mems]):
            for m in self.task_memids:
                # activate
                TaskNode(self.agent.memory, m).get_update_status({"prio": 1, "paused": 0})
        else:
            for m in self.task_memids:
                # mark tasks as finished
                TaskNode(self.agent.memory, m).get_update_status({"prio": -2})
        return


class ConfirmReferenceObject(Task):
    """This Task confirms if a reference object is correct.

    if there is a response that maps to "yes"  it will place a triple
    (subj=self.memid, pred_text="dialogue_task_output", obj_text="yes")
    else (either no response, or maps no, or otherwise)
    (subj=self.memid, pred_text="dialogue_task_output", obj_text="no")

    Args:
        bounds: general area of reference object to point at
        pointed: flag determining whether the agent pointed at the area
        asked: flag determining whether the confirmation was asked for

    """

    def __init__(self, agent, task_data={}):
        task_data["blocking"] = True
        super().__init__(agent, task_data=task_data)
        r = task_data.get("reference_object")
        if hasattr(r, "get_point_at_target"):
            self.bounds = r.get_point_at_target()
        else:
            # this should be an error
            self.bounds = tuple(np.min(r, axis=0)) + tuple(np.max(r, axis=0))
        self.pointed = False
        self.asked = False
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        """Confirm the block object by pointing and wait for answer."""
        if not self.asked:
            task_list = [
                Say(self.agent, {"response_options": self.question}),
                AwaitResponse(self.agent, {"asker_memid": self.memid}),
            ]
            self.add_child_task(task_list)
            self.asked = True
            return
        if not self.pointed:
            # FIXME agent shouldn't just point, should make a task etc.
            self.agent.point_at(self.bounds)
            self.add_child_task(AwaitResponse(self.agent))
            self.pointed = True
            return
        self.finished = True
        # FIXME: change this to sqly when syntax for obj searches is settled:
        t = self.agent.memory.get_triples(subj=self.memid, pred_text="dialogue_task_response")
        chat_mems = [self.agent.memory.get_mem_by_id(triples[2]) for triples in t]
        response = "no"
        if chat_mems:
            # FIXME...
            if chat_mems[0].chat_text in MAP_YES:
                response = "yes"
        self.agent.memory.add_triple(
            subj=self.memid, pred_text="dialogue_task_output", obj_text=response
        )
        return


def build_question_json(
    text: str,
    media: list = None,
    text_response_options: list = None,
    media_response_options: list = None,
):
    chat_obj = {  # based on the schema from droidlet/dialog/chat_schema.md{
        "chat_memid": "",
        "timestamp": 0,
        "content_type": "",
        "content": [],
    }
    if media_response_options:
        chat_obj["content_type"] = "chat_and_media_options"
    elif media and text_response_options:
        chat_obj["content_type"] = "chat_and_media_and_text_options"
    elif text_response_options:
        chat_obj["content_type"] = "chat_and_text_options"
    elif media:
        chat_obj["content_type"] = "chat_and_media"
    elif len(text) > 6 and text[:6] == "/point":
        chat_obj["content_type"] = "point"
    else:
        chat_obj["content_type"] = "chat_string"

    chat_obj["content"].append({"id": "text", "content": f"{text}"})
    if media:
        for m in media:
            chat_obj["content"].append({"id": "image_link", "content": f"{m}"})
    if text_response_options:
        for tro in text_response_options:
            chat_obj["content"].append({"id": "response_option", "content": f"{tro}"})
    if media_response_options:
        for mro in media_response_options:
            chat_obj["content"].append({"id": "response_image_link", "content": f"{mro}"})

    return json.dumps(chat_obj)


def check_parse(task):
    question = f"I'm not sure about something. I think you wanted me to {task.action.lower()} a {task.ref_obj_span}, is that right?"
    question_obj = build_question_json(question, text_response_options=["yes", "no"])
    task_list = [
        Say(task.agent, {"response_options": question_obj}),
        AwaitResponse(task.agent, {"asker_memid": task.memid}),
    ]
    task.add_child_task(maybe_task_list_to_control_block(task_list, task.agent))


def map_yes_last_chat(task):
    chat_mem = task.agent.memory.get_most_recent_incoming_chat(after=task.step_time + 1)
    response = "no"
    if chat_mem:
        # FIXME...
        if chat_mem.chat_text in MAP_YES:
            response = "yes"
    return response


def point_at(task, target):
    if hasattr(target, "get_point_at_target"):
        bounds = target.get_point_at_target()
    else:
        # FIXME is there a more graceful way to handle this?
        return
    # FIXME agent shouldn't just point, should make a task etc.
    task.agent.point_at(bounds)
    question = f"Is this the {task.ref_obj_span}? (Look for the flashing object)"
    task_list = [
        Say(task.agent, {"response_options": question}),
        AwaitResponse(task.agent, {"asker_memid": task.memid}),
    ]
    task.add_child_task(maybe_task_list_to_control_block(task_list, task.agent))
    task.asks += 1
    task.step_time = task.agent.memory.get_time()


def clarification_failed(task):
    question = "OK, I didn't understand you correctly.  Please mark this as an error."
    task.add_child_task(Say(task.agent, {"response_options": question}))
    task.asks = task.max_asks
    task.finished = True


class ClarifyCC1(Task):
    """This Task is responsible
    for Say and AwaitResponse tasks associated with clarifying which
    among an expanded list of candidates is the appropriate ref_obj

    Args:

    """

    def __init__(self, agent, memid, task_data={}):
        self.memid = memid
        task_data["blocking"] = True
        super().__init__(agent, task_data=task_data)
        self.dlf = task_data.get("dlf")
        self.candidates = self.dlf["class"]["candidates"]
        self.currect_candidate = None
        self.action = self.dlf["action"]["action_type"]
        self.ref_obj_span = self.dlf["action"]["reference_object"][
            "text_span"
        ]  # FIXME will this always be here?
        self.finished = False
        self.step_time = self.agent.memory.get_time()
        self.max_asks = len(self.candidates) + 1  # verify action + ref_obj span, then candidates
        self.asks = 1
        clarify_dlf_task = TaskNode(agent.memory, self.memid)
        clarify_dlf_task.update_task(task=self)
        clarify_dlf_task.get_update_status({"prio": 1})

    @Task.step_wrapper
    def step(self):
        """Issue chats and wait for responses to clarify"""

        print("ClarifyCC1 stepped")
        # import ipdb

        # ipdb.set_trace(context=7)

        if not self.finished and self.asks <= self.max_asks:
            if self.asks == 1:
                # ask whether the original parse is nominally right
                check_parse(self)
                self.asks += 1
                return

            elif self.asks == 2:
                response = map_yes_last_chat(self)
                if response == "yes":
                    # The parse was at least kind of right, start suggesting objects
                    self.currect_candidate = self.candidates.pop(0)
                    point_at(self, self.agent.memory.get_mem_by_id(self.currect_candidate))
                else:
                    # Seemingly a bad parse, move on to error marking
                    clarification_failed(self)
                return

            else:
                # Check if the last obj was right, if not continue
                response = map_yes_last_chat(self)
                if response == "no":
                    self.currect_candidate = self.candidates.pop(0)
                    point_at(self, self.agent.memory.get_mem_by_id(self.currect_candidate))
                else:
                    # Found it! Add the approriate tag to current candidate and mark it as the output
                    self.agent.memory.add_triple(
                        subj=self.currect_candidate,
                        pred_text="has_tag",
                        obj_text=self.ref_obj_span,
                    )
                    self.agent.memory.add_triple(
                        subj=self.memid,
                        pred_text="dialogue_clarification_output",
                        obj_text=self.currect_candidate,
                    )
                    self.add_child_task(
                        Say(self.agent, {"response_options": "Thank you for clarifying!"})
                    )
                    self.finished = True
                return

        else:
            # We ran out of candidates, move on to error marking
            clarification_failed(self)
            return
