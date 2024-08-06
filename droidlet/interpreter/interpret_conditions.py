"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Optional, Union
from word2number.w2n import word_to_num
from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.task.condition_classes import (
    Condition,
    NeverCondition,
    AndCondition,
    OrCondition,
    TimeCondition,
)
from .interpret_comparators import interpret_comparator


class ConditionInterpreter:
    def __call__(self, interpreter, speaker, d) -> Optional[Condition]:
        """subinterpreter for Conditions
        args:
        interpreter:  root interpreter.
        speaker (str): The name of the player/human/agent who uttered
            the chat resulting in this interpreter
        d: logical form from semantic parser
        """
        error_msg = "I thought there was a condition but I don't understand it:"
        # FIXME! there are some leftover incorrect "condition_type" keys in data
        if d.get("condition_type") is not None:
            if d["condition_type"] == "ALWAYS" or d["condition_type"] == "NEVER":
                d = {"condition": d["condition_type"]}
        if d.get("condition") is None:
            raise ErrorWithResponse(error_msg + " {}".format(d))
        d = d["condition"]
        if type(d) is str:
            if d == "NEVER":
                return NeverCondition(interpreter.memory)
            elif d == "ALWAYS":
                return AlwaysCondition(interpreter.memory)
            else:
                raise ErrorWithResponse(error_msg + " {}".format(d))
        assert type(d) is dict
        conds = []
        if d.get("and_condition"):
            for c in d["and_condition"]:
                conds.append(self(interpreter, speaker, c))
                return AndCondition(interpreter.memory, conds)
        elif d.get("or_condition"):
            for c in d["or_condition"]:
                conds.append(self(interpreter, speaker, c))
                return OrCondition(interpreter.memory, conds)
        elif d.get("input_left") or d.get("input_right"):
            return interpret_comparator(interpreter, speaker, d)
        else:
            # this condition should have a comparator.
            C = d.get("comparator", {})
            if not C:
                raise ErrorWithResponse(error_msg + " {}".format(d))
            # is it a time condition?
            if (
                d.get("special_time_event")
                or d.get("event")
                or C.get("input_left", "NULL") == "CURRENT_TIME"
            ):
                return self.interpret_time(interpreter, speaker, d)
            else:
                return interpret_comparator(interpreter, speaker, C)

    def interpret_time(self, interpreter, speaker, d):
        event = None
        if d.get("special_time_event"):
            return TimeCondition(interpreter.memory, d["special_time_event"])
        else:
            if not d.get("comparator"):
                raise ErrorWithResponse("I don't know how to interpret this time condition")
            dc = d["comparator"]
            # the Comparator's input_left is eventually going to be converted into a TimeValue; the
            # TimeValue will be specified in TimeCondition.  for now it does not need to
            # be interpeted in interpret_comparator.
            dc["input_left"] = "NULL"
            comparator = interpret_comparator(interpreter, speaker, dc)

        if d.get("event"):
            event = self(interpreter, speaker, d["event"])

        return TimeCondition(interpreter.memory, comparator, event=event)
