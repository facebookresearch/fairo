"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import datetime
from typing import Tuple, Dict, Any, Optional
from droidlet.event import dispatch

from .interpreter_utils import SPEAKERLOOK

# point target should be subinterpret, dance should be in agents subclassed interpreters
from .interpret_reference_objects import ReferenceObjectInterpreter, interpret_reference_object
from .interpret_location import ReferenceLocationInterpreter, interpret_relative_direction
from .interpret_filters import FilterInterpreter
from droidlet.shared_data_structs import ErrorWithResponse, NextDialogueStep
from droidlet.task.task import maybe_task_list_to_control_block
from droidlet.memory.memory_nodes import TripleNode, TaskNode, InterpreterNode
from droidlet.dialog.dialogue_task import ConfirmTask


class InterpreterBase:
    def __init__(
        self, speaker, logical_form_memid, agent_memory, memid=None, interpreter_type="interpreter"
    ):
        self.speaker = speaker  # TODO put this in memory
        self.memory = agent_memory
        if not memid:
            self.memid = InterpreterNode.create(self.memory, interpreter_type=interpreter_type)
            if (
                logical_form_memid != "NULL"
            ):  # if it were "NULL", this is a dummy interpreter of some sort...
                self.memory.add_triple(
                    subj=self.memid, pred_text="logical_form_memid", obj=logical_form_memid
                )
        else:
            # ALL state is stored in memory, associated to the interpreter memid.
            self.memid = memid
            logical_form_memids, _ = agent_memory.basic_search(
                "SELECT uuid FROM Program WHERE <<#{}, logical_form_memid, ?>>".format(memid)
            )
            if not logical_form_memids:
                raise Exception(
                    "tried to make an interpreter from memid but no logical form was associated to it"
                )
            if len(logical_form_memids) > 1:
                raise Exception("multiple logical forms associated to single interpreter memory")
            logical_form_memid = logical_form_memids[0]

        if (
            logical_form_memid != "NULL"
        ):  # if it were "NULL", this is a dummy interpreter of some sort...
            logical_form_mem = agent_memory.get_mem_by_id(logical_form_memid)
            self.logical_form = logical_form_mem.logical_form

    def step(self, agent):
        raise NotImplementedError()


class Interpreter(InterpreterBase):
    """
    | This class takes a logical form from the semantic parser that specifies a
    | (world affecting) action for the agent 
    | and the world state (from the agent's memory),
    | and uses these to intialize tasks to run
    | Most of the logic of the interpreter is run in the subinterpreters and task handlers.

    Args:
        speaker: The name of the player/human/agent who uttered the chat resulting in this interpreter
        action_dict: The logical form, e.g. returned by a semantic parser

    Keyword Args:
        agent: the agent running this Interpreter
        memory: the agent's memory
        dialogue_stack: a DialogueStack object where this Interpreter object will live
    """

    def __init__(self, speaker, logical_form_memid, agent_memory, memid=None):
        super().__init__(speaker, logical_form_memid, agent_memory, memid=memid)

        self.default_debug_path = "debug_interpreter.txt"
        self.post_process_loc = lambda loc, interpreter: loc

        # make sure to do in subclass
        # this is the order of things to search in workspace memory if a ref object
        # is not found, FIXME! (use more sophisticated search):
        self.workspace_memory_prio = []  # noqa

        # reorganize by signature?
        # e.g.
        #  logical_form --> list(mems)
        #  logical_form, list(mems) --> list(mems)
        #  logical_form, list(mems), list(vals) --> list(mems), list(vals)
        self.subinterpret = {
            "filters": FilterInterpreter(),
            # FIXME, just make a class
            "reference_objects": ReferenceObjectInterpreter(interpret_reference_object),
            "reference_locations": ReferenceLocationInterpreter(),
            # make sure to do this in subclass
            # "attribute": MCAttributeInterpreter(),
            # "condition": ConditionInterpreter(),
            # "specify_locations": ComputeLocations(),
            # "facing": FacingInterpreter(),
        }

        # each action handler should have signature
        # speaker, logical_form -->  a Task, possible output utterance, possible output data.
        #     if the action handler builds a list, it should wrap it in a ControlBlock
        self.action_handlers = {
            "MOVE": self.handle_move,
            "STOP": self.handle_stop,
            "RESUME": self.handle_resume,
            "UNDO": self.handle_undo,
            "OTHERACTION": self.handle_otheraction,
        }

        # fill these in subclasses
        self.task_objects = {}  # noqa

    def step(self, agent) -> Tuple[Optional[str], Any]:
        start_time = datetime.datetime.now()
        assert self.logical_form["dialogue_type"] == "HUMAN_GIVE_COMMAND"
        try:
            actions = []
            if "action" in self.logical_form:
                actions.append(self.logical_form["action"])
            elif "action_sequence" in self.logical_form:
                actions = self.logical_form["action_sequence"]

            if len(actions) == 0:
                # The action dict is in an unexpected state
                raise ErrorWithResponse(
                    "I thought you wanted me to do something, but now I don't know what"
                )
            tasks_to_push = []
            for action_def in actions:
                action_type = action_def["action_type"]
                r = self.action_handlers[action_type](agent, self.speaker, action_def)
                if len(r) == 3:
                    task, response, dialogue_data = r
                else:
                    # FIXME don't use this branch, uniformize the signatures
                    task = None
                    response, dialogue_data = r
                if task:
                    tasks_to_push.append(task)
            task_mem = None
            if tasks_to_push:
                T = maybe_task_list_to_control_block(tasks_to_push, agent)
                #                task_mem = TaskNode(self.memory, tasks_to_push[0].memid)
                task_mem = TaskNode(self.memory, T.memid)
            if task_mem:
                chat = self.memory.get_most_recent_incoming_chat()
                TripleNode.create(
                    self.memory, subj=chat.memid, pred_text="chat_effect_", obj=task_mem.memid
                )
            self.finished = True
            end_time = datetime.datetime.now()
            hook_data = {
                "name": "interpreter",
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_time": (end_time - start_time).total_seconds(),
                "agent_time": self.memory.get_time(),
                "tasks_to_push": tasks_to_push,
                "task_mem": task_mem,
            }
            dispatch.send("interpreter", data=hook_data)
        except NextDialogueStep:
            return
        except ErrorWithResponse as err:
            self.finished = True
        return

    def handle_undo(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        Undo = self.task_objects["undo"]
        task_name = d.get("undo_action")
        if task_name:
            task_name = task_name.split("_")[0].strip()
        old_task = self.memory.get_last_finished_root_task(task_name)
        if old_task is None:
            raise ErrorWithResponse("Nothing to be undone ...")
        undo_tasks = [Undo(agent, {"memid": old_task.memid})]
        for u in undo_tasks:
            agent.memory.get_mem_by_id(u.memid).get_update_status({"paused": 1})
        undo_command = old_task.get_chat().chat_text

        logging.debug("Pushing ConfirmTask tasks={}".format(undo_tasks))
        confirm_data = {
            "task_memids": [u.memid for u in undo_tasks],
            "question": 'Do you want me to undo the command: "{}" ?'.format(undo_command),
        }
        ConfirmTask(agent, confirm_data)
        self.finished = True
        return None, None

    def handle_move(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        Move = self.task_objects["move"]
        Control = self.task_objects["control"]

        loop_mem = None
        if "remove_condition" in d:
            condition = self.subinterpret["condition"](self, speaker, d["remove_condition"])
            location_d = d.get("location", SPEAKERLOOK)
            mems = self.subinterpret["reference_locations"](self, speaker, location_d)
            if mems:
                loop_mem = mems[0]

        def new_tasks():
            # TODO if we do this better will be able to handle "stay between the x"
            default_loc = getattr(self, "default_loc", SPEAKERLOOK)
            location_d = d.get("location", default_loc)
            # FIXME, this is hacky.  need more careful way of storing this in task
            if loop_mem:
                mems = [loop_mem]
            else:
                mems = self.subinterpret["reference_locations"](self, speaker, location_d)
            # FIXME this should go in the ref_location subinterpret:
            steps, reldir = interpret_relative_direction(self, location_d)
            pos, _ = self.subinterpret["specify_locations"](self, speaker, mems, steps, reldir)
            # TODO: can this actually happen?
            if pos is None:
                raise ErrorWithResponse("I don't understand where you want me to move.")
            pos = self.post_process_loc(pos, self)
            task_data = {"target": pos, "action_dict": d}
            task = Move(agent, task_data)
            return task

        if "remove_condition" in d:
            # FIXME grammar to handle "remove" vs "stop"
            loop_task_data = {
                "new_tasks": new_tasks,
                "remove_condition": condition,  #!!! semantic parser + GT need updating
                "action_dict": d,
            }
            return Control(agent, loop_task_data), None, None
        else:
            return new_tasks(), None, None

    # TODO mark in memory it was stopped by command
    def handle_stop(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.memory.task_stack_pause():
            return None, "Stopping.  What should I do next?", None
        else:
            return None, "I am not doing anything", None

    # FIXME this is needs updating...
    # TODO mark in memory it was resumed by command
    def handle_resume(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.memory.task_stack_resume():
            return None, "resuming", None
        else:
            return None, "nothing to resume", None

    def handle_otheraction(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        return None, "I don't know how to do that yet", None
