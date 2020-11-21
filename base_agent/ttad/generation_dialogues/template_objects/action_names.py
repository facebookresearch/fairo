"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated directly with Action names.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *


class Dance(TemplateObject):
    """This template object repesents a single word 'Dance' command.
    eg: dance / dance around"""

    def add_generate_args(self, index=0, templ_index=0):
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1

        if template_len == 1:
            self.node._no_children = True

        single_commands = [
            ["dance", "dance"],
            ["dance", "do a dance"],
            ["dance", "show me a dance"],
        ]
        all_names = get_template_names(self, templ_index)
        template_len = len(all_names) - 1

        if template_len == 2 and (all_names[-1] == "ConditionTypeNever"):
            single_commands = [
                ["dancing", "keep dancing"],
                ["dance", "dance forever"],
                ["dance", "dance until I tell you to stop"],
            ]

        command = random.choice(single_commands)
        self.node.dance_type_name = command[0]
        self.node._dance_text = command[1]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = self.node._dance_text
        prefix = random.choice(["", random.choice(["can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()

        return new_command


class Move(TemplateObject):
    """This template object repesents the 'Move' command"""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["move", "go", "come", "walk"])
        all_names = get_template_names(self, templ_index)

        # If next argument is "Next", 'go next to X'
        if all_names[1] == "Next" or ("Between" in all_names):
            command = "go"

        # for follow ups on a previous move command
        if "HumanReplace" in all_names:
            return None
        # Infinite loop
        if "ConditionTypeNever" in all_names and any(
            x in ["LocationMobTemplate", "BlockObjectIt", "BlockObjectThat"] for x in all_names
        ):
            command = random.choice(["follow", "catch", "keep following"])
        elif "ClimbDirectionTemplate" in all_names:
            command = random.choice(["climb"])
        else:
            # for away, use 'move'
            description = self.node.args[0].generate_description()
            if "relative_direction" in description:
                if all_names[-1] != "ConditionTypeAdjacentBlockType":
                    if "away" in description["relative_direction"]:
                        command = "move"
                else:
                    command = random.choice(["keep moving", "keep walking", "walk"])
            elif "StepsTemplate" in all_names:  # move X steps
                command = random.choice(["move", "walk"])

        prefix = random.choice(["", random.choice(["", "can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()

        return new_command


class MoveSingle(TemplateObject):
    """This template object repesents a single word 'Move' command.
    eg: move / move somewhere"""

    def add_generate_args(self, index=0, templ_index=0):
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1
        if template[0] == "Human" and template_len == 1:
            self.node._no_children = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        single_commands = ["move", "walk"]
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1

        if template[0] == "Human" and template_len == 1:
            enhancements = ["anywhere", "somewhere", "around"]
            new_commands = []
            for comm in enhancements:
                for command in single_commands:
                    new_commands.append(" ".join([command, comm]))
            single_commands.extend(new_commands)

        elif template_len == 2 and (template[-1] == "ConditionTypeNever"):
            single_commands = ["keep walking", "keep moving"]

        command = random.choice(single_commands)
        prefix = random.choice(["", random.choice(["can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()

        return new_command


### BUILD ###


class Build(TemplateObject):
    """This template object represents the 'Build' command"""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)

        if "HumanReplace" in template_names:
            return None
        command_list = [
            "build",
            "make",
            "construct",
            "assemble",
            "create",
            "rebuild",
            "install",
            "place",
            "put",
        ]
        replace_flag = True if "HumanReplace" in template_names else False

        if not replace_flag:
            command_list.extend(["build me", "make me"])

        command = random.choice(command_list)
        if not replace_flag:
            prefix = random.choice(
                ["", random.choice(["", "can you", "please", "can you please"])]
            )
            if command in ["build me", "make me"]:
                prefix = random.choice(
                    ["", random.choice(["", "can you", "please", "can you please"])]
                )

        new_command = (
            random.choice([(" ".join([prefix, command])).strip(), "we need"])
            if not replace_flag
            else command
        )

        return new_command


class BuildSingle(TemplateObject):
    """This template object represents single word (no arguments) 'Build' command"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._no_children = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "build",
                "make",
                "construct",
                "assemble",
                "create",
                "build something",
                "build me something",
                "make something",
                "make something for me",
                "make me something",
                "construct something",
                "assemble something",
                "create something",
                "build anything",
                "build me anything",
                "make anything",
                "make me anything",
                "construct anything",
                "assemble anything",
                "create anything",
                "build something you know",
                "build me something you know",
                "make something you know",
                "make me something you know",
                "construct something you know",
                "assemble something you know",
                "create something you know",
                "build anything you know",
                "make anything you know",
                "construct anything you know",
                "assemble anything you know",
                "create anything you know",
                "build stuff",
                "build me stuff",
                "make stuff",
                "create stuff",
                "install something",
                "install stuff",
                "install something for me please",
            ]
        )

        prefix = random.choice(["", random.choice(["", "can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()
        return new_command


### DIG ###


class Dig(TemplateObject):
    """This template object repesents the Dig command"""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template = self.node.template[templ_index]
        template_names = get_template_names(self, templ_index)
        command = random.choice(["dig", "mine", "clear"])

        if (
            "DigSomeShape" in template_names
            and type(template[template_names.index("DigSomeShape")]._child).__name__
            == "DigShapeHole"
        ):
            command = random.choice(["make", "build"])
        prefix = random.choice(
            ["", random.choice(["", "can you", "please", "can you please", "let 's", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()

        return new_command


class DigSingle(TemplateObject):
    """This template object repesents single word Dig command with no arguments"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._no_children = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "dig",
                "mine",
                "dig something",
                "mine something",
                "dig anything",
                "mine anything",
                "dig stuff",
                "make a hole",
            ]
        )
        prefix = random.choice(
            ["", random.choice(["", "can you", "please", "can you please", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()

        return new_command


### FREEBUILD ###


class Freebuild(TemplateObject):
    """This template object repesents a Freebuild command"""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "complete",
                "can you please complete",
                "please complete",
                "can you complete",
                "finish building",
            ]
        )
        return command


class FreebuildLocation(TemplateObject):
    """This template object repesents a Freebuild command with only Location"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._only_location = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "help me build something",
                "help me make something",
                "can you help me build something",
                "can you help me make something",
                "can you please help me build something",
                "help me make something",
                "help me make",
                "help me build",
                "build something with me",
                "make something with me",
                "let 's build something together",
                "let 's build something",
            ]
        )
        return command


### DESTROY ###


class Destroy(TemplateObject):
    """This template object repesents the Destroy/ Destroy command"""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "destroy",
                "destroy",
                "remove",
                "destruct",
                "knock down",
                "explode",
                "blow up",
                "tear down",
                "dismantle",
                "cut down",
                "chop",
                "clear",
                "chop down",
            ]
        )
        prefix = random.choice(
            ["", random.choice(["can you", "please", "can you please", "let 's", "help me"])]
        )
        new_command = random.choice([(" ".join([prefix, command])).strip(), "i do n't want"])

        return new_command


class DestroySingle(TemplateObject):
    """This template object repesents single word Destroy command with no arguments"""

    def add_generate_args(self, index=0, templ_index=0):
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1

        if template_len == 1:
            self.node._no_children = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        single_commands = ["destroy", "remove", "destruct", "knock down", "explode", "dismantle"]
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1

        if template_len == 1:
            enhancements = ["something", "anything"]
            new_commands = []
            for comm in enhancements:
                for command in single_commands:
                    new_commands.append(" ".join([command, comm]))
            single_commands.extend(new_commands)
        elif template_len == 2 and template[-1] == "RepeatAll":
            single_commands.extend(["clear", "chop down"])
            enhancements = [
                "everything",
                "everything around",
                "everything until I ask you to stop",
                "everything until I tell you to stop",
            ]
            new_commands = []
            for comm in enhancements:
                for command in single_commands:
                    new_commands.append(" ".join([command, comm]))
            new_commands.extend(["clear area", "clear the whole area"])
            single_commands = new_commands

        command = random.choice(single_commands)
        prefix = random.choice(
            ["", random.choice(["can you", "please", "can you please", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()

        return new_command


### SPAWN ###


class Spawn(TemplateObject):
    """This template object repesents the Spawn command."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["spawn", "create", "produce", "generate"])
        prefix_choice = ["can you", "please", "can you please", "help me"]
        prefix = random.choice(["", random.choice(prefix_choice)])
        command = (" ".join([prefix, phrase])).strip()
        return command


### FILL ###


class Fill(TemplateObject):
    """This template object repesents the Fill command"""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        phrases = ["fill", "cover"]

        if "Up" not in template_names:
            phrases.extend(["fill up", "cover up"])

        phrase = random.choice(phrases)
        prefix = random.choice(
            ["", random.choice(["can you", "please", "can you please", "help me"])]
        )
        command = (" ".join([prefix, phrase])).strip()

        return command


### UNDO ###


class Undo(TemplateObject):
    """This template object repesents the Undo / revert action """

    def add_generate_args(self, index=0, templ_index=0):
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1

        if template_len == 1:
            self.node._no_children = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrases = ["undo", "revert"]
        template = get_template_names(self, templ_index)
        template_len = len(template) - 1

        if template_len == 1:
            phrases.extend(
                ["undo what you just did", "undo last action", "revert last action", "undo that"]
            )
        phrase = random.choice(phrases)
        prefix = random.choice(
            ["", random.choice(["can you", "please", "can you please", "help me"])]
        )
        command = (" ".join([prefix, phrase])).strip()

        return command


### STOP ###


class StopSingle(TemplateObject):
    """This template object repesents that the action that needs to be undone is
    Build."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrases = [
            "stop",
            "hold on",
            "wait",
            "pause",
            "stop doing that",
            "stop doing what you are doing",
            "stop what you are doing",
            "do n't do that",
            "stop task",
        ]
        return random.choice(phrases)


class Stop(TemplateObject):
    """This template object repesents that the action that needs to be undone is
    Build."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return random.choice(["stop", "stay"])


### RESUME ###


class ResumeSingle(TemplateObject):
    """This template object repesents that the action that needs to be undone is
    Build."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrases = [
            "resume",
            "continue",
            "restart",
            "start again",
            "keep going on",
            "keep going",
            "keep doing that",
            "keep doing what you were doing",
            "continue doing that",
            "continue doing what you were doing",
            "continue what you were doing",
            "go back to doing what you were doing",
            "go back to what you were doing",
        ]
        return random.choice(phrases)


class Resume(TemplateObject):
    """This template object repesents that the action that needs to be undone is
    Build."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return random.choice(["resume", "keep", "continue"])


### COPY ###

"""This template object represents the Copy action."""


class Copy(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        replace_flag = True if "HumanReplace" in template_names else False

        if any(x in ["RepeatAll", "RepeatCount"] for x in template_names):
            command = random.choice(
                [
                    "make copies of",
                    "make me copies of",
                    "create copies of",
                    "copy",
                    "replicate",
                    "reproduce",
                    "emulate",
                    "make another of",
                ]
            )
        else:
            template_names = get_template_names(self, templ_index)

            command_list = ["copy", "replicate", "reproduce", "emulate"]

            if not replace_flag:
                command_list.extend(
                    [
                        "make a copy of",
                        "make me a copy of",
                        "create a copy of",
                        "make copy of",
                        "make another one of",
                        "build me another one of",
                    ]
                )

            command = random.choice(command_list)
        if not replace_flag:
            prefix = random.choice(
                ["", random.choice(["", "can you", "please", "can you please", "help me"])]
            )

        new_command = (
            random.choice([(" ".join([prefix, command])).strip()]) if not replace_flag else command
        )

        return new_command


class CopyMultiple(TemplateObject):
    """This template object represents the Copy action where mutiple copies need to be
    made."""

    def add_generate_args(self, index=0, templ_index=0):
        num_copies = random.choice(self.template_attr.get("count", range(1, 101)))
        self.num_copies = random.choice(
            [str(num_copies), int_to_words(num_copies), "a few", "some"]
        )
        self.node._repeat_args["repeat_key"] = "FOR"
        self.node._repeat_args["repeat_count"] = self.num_copies

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "make {} copies of",
                "make me {} copies of",
                "create {} copies of",
                "make {} of",
                "make me {} of",
                "create {} of",
            ]
        ).format(self.num_copies)

        prefix = random.choice(
            ["", random.choice(["", "can you", "please", "can you please", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()

        return new_command


"""This template object represents a single word Copy action with no arguments."""


class CopySingle(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._no_children = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(
            [
                "copy",
                "make a copy",
                "make me a copy",
                "create a copy",
                "copy something",
                "make a copy of something",
                "create a copy of something",
                "copy anything",
                "make a copy of anything",
                "create a copy of anything",
            ]
        )
        prefix = random.choice(
            ["", random.choice(["", "can you", "please", "can you please", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()
        return new_command


### TAG ###

"""This template object represents the Tag action."""


class Tag(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["tag", "label", "name"])
        prefix = random.choice(
            ["", random.choice(["", "can you", "please", "can you please", "let 's", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()
        return new_command
