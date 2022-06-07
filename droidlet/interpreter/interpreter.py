"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import datetime
from copy import deepcopy
from typing import Tuple, Dict, Any, Optional
from droidlet.event import dispatch

from .interpreter_utils import SPEAKERLOOK

# point target should be subinterpret, dance should be in agents subclassed interpreters
from .interpret_reference_objects import ReferenceObjectInterpreter, interpret_reference_object
from .interpret_location import ReferenceLocationInterpreter, interpret_relative_direction
from .interpret_filters import FilterInterpreter
from droidlet.shared_data_structs import ErrorWithResponse, NextDialogueStep
from droidlet.task.task import task_to_generator, ControlBlock
from droidlet.memory.memory_nodes import ChatNode, TripleNode, InterpreterNode
from droidlet.dialog.dialogue_task import ConfirmTask, Say


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
                self.memory.nodes[TripleNode.NODE_TYPE].create(
                    self.memory,
                    subj=self.memid,
                    pred_text="logical_form_memid",
                    obj=logical_form_memid,
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

        if logical_form_memid != "NULL":
            # if it were "NULL", this is a dummy interpreter of some sort... probably from a unit test
            self.logical_form_memid = logical_form_memid
            logical_form_mem = agent_memory.get_mem_by_id(self.logical_form_memid)
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
        logical_form_memid:  pointer to the parse to be interpreted
        agent_memory:  memory of the agent that will interpret the parse
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
        # speaker, logical_form --> callable that returns Tasks
        # it can place dialogue tasks as a side effect.
        # TODO remove dialogue task side effects....
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
        self.finished = False
        try:
            C = self.interpret_event(agent, self.speaker, self.logical_form)
            if C is not None:
                chat = self.memory.nodes[ChatNode.NODE_TYPE].get_most_recent_incoming_chat(
                    self.memory
                )
                TripleNode.create(
                    self.memory, subj=chat.memid, pred_text="chat_effect_", obj=C.memid
                )
            self.finished = True
            end_time = datetime.datetime.now()

            # FIXME (tasks to push)
            task_memid = "NULL"
            if C is not None:
                task_memid = C.memid
            hook_data = {
                "name": "interpreter",
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_time": (end_time - start_time).total_seconds(),
                "agent_time": self.memory.get_time(),
                "tasks_to_push": [],
                "task_mem": task_memid,
            }
            dispatch.send("interpreter", data=hook_data)
        except NextDialogueStep:
            return
        except ErrorWithResponse as err:
            Say(agent, task_data={"response_options": err.chat})
            self.finished = True
        return

    def interpret_event(self, agent, speaker, d):
        """
        recursively interpret an EVENT sub-lf
        this returns new_tasks callable if it operates on a leaf
        and a ControlBlock otherwise.
        """
        if "action_type" in d:
            action_type = d["action_type"]
            assert "event_sequence" not in d
            return self.action_handlers[action_type](agent, speaker, d)
        else:
            assert "event_sequence" in d
            terminate_condition = None
            init_condition = None
            if "terminate_condition" in d:
                terminate_condition = self.subinterpret["condition"](
                    self, speaker, d["terminate_condition"]
                )
            if "init_condition" in d:
                init_condition = self.subinterpret["condition"](self, speaker, d["init_condition"])
            task_gens = []
            for e_lf in d["event_sequence"]:
                task_gen = self.interpret_event(agent, speaker, e_lf)
                # FIXME: better handling of empty task_gens, e.g. from undo
                if task_gen is not None:
                    if not callable(task_gen):
                        # this should be a ControlBlock...
                        assert type(task_gen) is ControlBlock
                        task_gen = task_to_generator(task_gen)
                    task_gens.append(task_gen)
            if task_gens:
                loop_task_data = {
                    "new_tasks": task_gens,
                    "init_condition": init_condition,
                    "terminate_condition": terminate_condition,
                    "action_dict": d,
                }
                return ControlBlock(agent, loop_task_data)
            else:
                return None

    # FIXME! undo is currently inaccurate due to ControlBlocks
    def handle_undo(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        Undo = self.task_objects["undo"]
        task_name = d.get("undo_action")
        if task_name:
            task_name = task_name.split("_")[0].strip()
        old_task = self.memory.get_last_finished_root_task(task_name, ignore_control=True)
        if old_task is None:
            raise ErrorWithResponse("Nothing to be undone ...")
        undo_tasks = [Undo(agent, {"memid": old_task.memid})]
        for u in undo_tasks:
            agent.memory.get_mem_by_id(u.memid).get_update_status({"paused": 1})
        try:
            undo_command = old_task.get_chat().chat_text
        except:
            # if parent was a ControlBlock, need to get chat from that
            old_task_parent = self.memory.get_last_finished_root_task(task_name)
            undo_command = old_task_parent.get_chat().chat_text
        logging.debug("Pushing ConfirmTask tasks={}".format(undo_tasks))
        confirm_data = {
            "task_memids": [u.memid for u in undo_tasks],
            "question": 'Do you want me to undo the command: "{}" ?'.format(undo_command),
        }
        ConfirmTask(agent, confirm_data)
        self.finished = True
        return None

    def handle_move(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        Move = self.task_objects["move"]

        def new_tasks():
            # TODO if we do this better will be able to handle "stay between the x"
            default_loc = getattr(self, "default_loc", SPEAKERLOOK)
            location_d = d.get("location", default_loc)
            # FIXME, this is hacky.  need more careful way of storing this in task
            # and to pass to task generator
            try:
                mems = self.subinterpret["reference_locations"](self, speaker, location_d)
            except NextDialogueStep:
                # TODO allow for clarification
                raise ErrorWithResponse("I don't understand where you want me to move.")
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

        return new_tasks

    # TODO mark in memory it was stopped by command
    def handle_stop(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.memory.task_stack_pause():
            Say(agent, task_data={"response_options": "Stopping.  What should I do next?"})
        return None

    # FIXME this is needs updating...
    # TODO mark in memory it was resumed by command
    def handle_resume(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.memory.task_stack_resume():
            Say(agent, task_data={"response_options": "Resuming"})
        else:
            Say(agent, task_data={"response_options": "nothing to resume"})

    def handle_otheraction(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        Say(agent, task_data={"response_options": "I don't know how to do that yet"})
