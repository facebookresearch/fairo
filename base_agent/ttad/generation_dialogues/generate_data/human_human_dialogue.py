"""
Copyright (c) Facebook, Inc. and its affiliates.

Actions:
- Move (optional<Location>, optional<StopCondition>, optional<Repeat>)
- Build (optional<Schematic>, optional<Location>, optional<Repeat>)
- Destroy (optional<BlockObject>)
- Dig (optional<has_length>, optional<has_width>, optional<has_depth>,
       optional<has_size>, optional<Location>, optional<StopCondition>,
       optional<Repeat>)
- Copy (optional<BlockObject>, optional<Location>, optional<Repeat>)
    - Undo (optional<target_action_type>)
- Fill (optional<Location>, optional<Repeat>)
- Spawn (Mob, optional<Repeat>)
- Freebuild (optional<BlockObject>, optional<Location>)
- Stop ()
- Resume ()
- Noop ()

Top-Level = {
    "dialogue_type": {
        "action_name" : {Action}
        }
    }
    e.g. {
    "human_give_command": {
        "Move" : {"Location": {Location}}
        }
    }

Action = {
    {arg_type}: {arg_dict}   # e.g. Move dict = {"Location": {Location}}
}
"""
from collections import OrderedDict

from .action_node import *
from generate_utils import *

from template_objects import (
    LOCATION_TEMPLATES,
    CONDIITON_TEMPLATES,
    REPEAT_KEY_TEMPLATES,
    BLOCKOBJECT_TEMPLATES,
)
from tree_components import Location, Schematic, BlockObject, Object, Mob, StopCondition, Repeat


############
## ACTION ##
############


class Move(ActionNode):
    """The Move action is used to move/ walk to a certain location. The action
    needs a Location to move to.
    __init__(): Pick a template from move_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Move", template, template_attr=template_attr)

    @classmethod
    def generate(cls, template=None, template_attr={}):
        move_obj = Move(template, template_attr)
        template = move_obj.template

        move_obj.ARG_TYPES = []
        move_obj._no_children = False  # no ARG_TYPE if _no_children is True
        move_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "coref_resolve": None,
                "relative_direction_value": None,
                "bo_coref_resolve": None,
                "template_attr": template_attr,
            }
        )

        move_obj._condition_args = Arguments({"condition_type": None, "block_type": None})
        # the number of repetitions for Action if any
        move_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        move_obj.args = []
        if move_obj._no_children:
            return move_obj

        move_obj.ARG_TYPES.append(Location)
        move_obj.args.append(Location(**move_obj._location_args))

        # StopCondition is optional, only add if the default arguments were changed.
        if move_obj._condition_args.values_updated:
            move_obj.ARG_TYPES.append(StopCondition)
            move_obj.args.append(StopCondition(**move_obj._condition_args))

        # Repeat is optional, only add if default values were updated
        if move_obj._repeat_args.values_updated:
            move_obj.ARG_TYPES.append(Repeat)
            move_obj.args.append(Repeat(**move_obj._repeat_args))

        return move_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            for i, key in enumerate(templ):
                key_type = type(key)
                arg_index = 0
                # arg_index 1 for StopCondition
                if (key_type in CONDIITON_TEMPLATES) and (len(self.args) > 1):
                    arg_index = 1

                # get the text from template object
                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Build(ActionNode):
    """The Build action is used to build something. The action needs a Schematic
    and maybe a Location to build something.
    __init__(): Pick a template from build_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Build", template, template_attr=template_attr)

    @classmethod
    def generate(cls, template=None, template_attr={}):
        build_obj = Build(template, template_attr)
        template = build_obj.template

        build_obj.ARG_TYPES = []
        build_obj._no_children = False  # no ARG_TYPE if _no_children is True
        build_obj._schematics_args = Arguments(
            {
                "only_block_type": False,
                "block_type": False,
                "schematic_attributes": False,
                "schematic_type": None,
                "abstract_size": None,
                "colour": None,
                "repeat_key": None,
                "repeat_dir": None,
                "template_attr": template_attr,
                "multiple_schematics": False,
            }
        )
        build_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "repeat_key": None,
                "coref_resolve": None,
                "template_attr": template_attr,
            }
        )

        # the number of repetitions for Action if any
        build_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        build_obj.args = []

        if build_obj._schematics_args.values_updated or not build_obj._no_children:
            build_obj.ARG_TYPES.append(Schematic)
            build_obj.args.append(Schematic(**build_obj._schematics_args))

        # Location is optional, only add if the default arguments were changed.
        if build_obj._location_args.values_updated:
            build_obj.ARG_TYPES.append(Location)
            build_obj.args.append(Location(**build_obj._location_args))

        # Repeat is optional, only add if default values were updated
        if build_obj._repeat_args.values_updated:
            build_obj.ARG_TYPES.append(Repeat)
            build_obj.args.append(Repeat(**build_obj._repeat_args))

        return build_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            for i, key in enumerate(templ):
                item = None
                key_type = type(key)

                arg_index = 0
                # arg_index 1 for Location
                if ((key_type in LOCATION_TEMPLATES) or (key_type in BLOCKOBJECT_TEMPLATES)) and (
                    len(self.args) > 1
                ):
                    arg_index = 1

                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue

                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                # shape_attributes can be a list
                elif type(item) == list:
                    result.extend(item)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Copy(ActionNode):
    """The Copy action is used to make a copy of something. The action is just the
    Build action with a BlockObject and maybe a Location to make the copy at.
    __init__(): Pick a template from copy_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Copy", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        copy_obj = Copy(template, template_attr)
        template = copy_obj.template

        copy_obj.ARG_TYPES = []
        copy_obj._no_children = False  # no ARG_TYPE if _no_children is True

        copy_obj._block_obj_args = Arguments(
            {
                "block_object_type": Object,
                "block_object_attributes": None,
                "block_object_location": False,
                "no_child": False,
                "repeat_key": None,
                "repeat_location": None,
                "coref_type": None,
                "template_attr": template_attr,
            }
        )
        copy_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "repeat_key": None,
                "template_attr": template_attr,
            }
        )
        # the number of repetitions for Action if any
        copy_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        copy_obj.args = []
        if copy_obj._no_children:
            return copy_obj

        if copy_obj._block_obj_args.values_updated:
            copy_obj.ARG_TYPES.append(BlockObject)
            copy_obj.args.append(BlockObject(**copy_obj._block_obj_args))

        # Location is optional, only add if the default arguments were changed.
        if copy_obj._location_args.values_updated:
            copy_obj.ARG_TYPES.append(Location)
            copy_obj.args.append(Location(**copy_obj._location_args))

        # Repeat is optional, only add if default values were updated
        if copy_obj._repeat_args.values_updated:
            copy_obj.ARG_TYPES.append(Repeat)
            copy_obj.args.append(Repeat(**copy_obj._repeat_args))

        return copy_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []

            for i, key in enumerate(templ):
                item = None
                key_type = type(key)
                arg_index = 0

                # check template_objects.py for the list of template objects
                if key_type in LOCATION_TEMPLATES:
                    if BlockObject in self.ARG_TYPES:
                        arg_index = 1
                elif key_type in REPEAT_KEY_TEMPLATES:
                    if Repeat in self.ARG_TYPES:
                        if BlockObject in self.ARG_TYPES:
                            if Location in self.ARG_TYPES:
                                arg_index = 2
                            else:
                                arg_index = 1

                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Dig(ActionNode):
    """The Dig action is used to dig something. The action needs a length, width
    and depth and maybe a Location to dig something at.
    __init__(): Pick a template from dig_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__(template_key="Dig", template=template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        dig_obj = Dig(template, template_attr)
        template = dig_obj.template

        dig_obj.ARG_TYPES = []
        dig_obj._no_children = False  # no ARG_TYPE if _no_children is True
        dig_obj.schematic = {}
        dig_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "repeat_key": None,
                "coref_resolve": None,
                "template_attr": template_attr,
            }
        )
        dig_obj._condition_args = Arguments({"condition_type": None, "block_type": None})
        # the number of repetitions for Action if any
        dig_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        dig_obj.args = []
        if dig_obj._no_children:
            return dig_obj

        # Location is optional, only add if the default arguments were changed.
        if dig_obj._location_args.values_updated:
            dig_obj.ARG_TYPES.append(Location)
            dig_obj.args.append(Location(**dig_obj._location_args))

        # StopCondition is optional, only add if the default arguments were changed.
        if dig_obj._condition_args.values_updated:
            dig_obj.ARG_TYPES.append(StopCondition)
            dig_obj.args.append(StopCondition(**dig_obj._condition_args))

        # Repeat is optional, only add if default values were updated
        if dig_obj._repeat_args.values_updated:
            dig_obj.ARG_TYPES.append(Repeat)
            dig_obj.args.append(Repeat(**dig_obj._repeat_args))

        return dig_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            for i, key in enumerate(templ):
                item = None
                key_type = type(key)

                arg_index = 0
                # arg_index 1 for StopCondition
                if key_type in CONDIITON_TEMPLATES and Location in self.ARG_TYPES:
                    arg_index = 1

                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)
                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Destroy(ActionNode):
    """The Destroy action is used to destroy something. The action needs a
    BlockObject to destroy.
    __init__(): Pick a template from destroy_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'block_object_type','block_object_attributes' etc. for BlockObject
    _generate_description(): Generates the text description using the template objects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Destroy", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        destroy_obj = Destroy(template, template_attr)
        template = destroy_obj.template

        destroy_obj.ARG_TYPES = []
        destroy_obj._no_children = False  # no ARG_TYPE if _no_children is True
        destroy_obj._block_obj_args = Arguments(
            {
                "block_object_type": Object,
                "block_object_attributes": [],
                "block_object_location": False,
                "no_child": False,
                "repeat_key": None,
                "repeat_no_child": None,
                "repeat_location": None,
                "coref_type": None,
                "template_attr": template_attr,
            }
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        destroy_obj.args = []
        if destroy_obj._no_children:
            return destroy_obj

        if destroy_obj._block_obj_args.values_updated:
            destroy_obj.ARG_TYPES.append(BlockObject)
            destroy_obj.args.append(BlockObject(**destroy_obj._block_obj_args))

        return destroy_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []

            for key in templ:
                key_type = type(key)

                arg_index = 0
                if key_type in CONDIITON_TEMPLATES and (len(self.args) > 1):
                    arg_index = 1

                item = key.generate_description(arg_index=arg_index, templ_index=j)

                if not item:
                    continue
                # flatten if nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Undo(ActionNode):
    """Undo action is used to revert an action/ last action.
    __init__(): Pick a template from undo_templates.py.
    generate(): Instantiates the template_objects in the template.
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Undo", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        undo_obj = Undo(template, template_attr)
        template = undo_obj.template

        undo_obj.ARG_TYPES = []
        undo_obj._no_children = False  # no ARG_TYPE if _no_children is True
        undo_obj.target_action_type = None  # name of action to be undone
        undo_obj.args = []

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        return undo_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            arg_index = 0

            for i, key in enumerate(templ):
                item = None
                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Fill(ActionNode):
    """Fill action is used to fill up holes. This action may have
    an optional location.
    __init__(): Pick a template from fill_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Fill", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        fill_obj = Fill(template, template_attr)
        template = fill_obj.template

        fill_obj.ARG_TYPES = []
        fill_obj._no_children = False  # no ARG_TYPE if _no_children is True
        fill_obj.has_block_type = None
        fill_obj.reference_object = {}
        fill_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "coref_resolve": None,
                "template_attr": template_attr,
            }
        )
        # the number of repetitions for Action if any
        fill_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        fill_obj.args = []
        if fill_obj._no_children:
            return fill_obj

        # Location is optional, only add if the default arguments were changed.
        if fill_obj._location_args.values_updated:
            fill_obj.ARG_TYPES.append(Location)
            fill_obj.args.append(Location(**fill_obj._location_args))

        # Repeat is optional, only add if default values were updated
        if fill_obj._repeat_args.values_updated:
            fill_obj.ARG_TYPES.append(Repeat)
            fill_obj.args.append(Repeat(**fill_obj._repeat_args))

        return fill_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            arg_index = 0

            for i, key in enumerate(templ):
                item = None
                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Spawn(ActionNode):
    """The Spawn action spawns a mob in the environment. The class needs a Mob to spawn.
    __init__(): Picks a template from templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of 'mob_location',
                'repeat_location' etc for Mob
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Spawn", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        spawn_obj = Spawn(template, template_attr)
        template = spawn_obj.template

        spawn_obj.ARG_TYPES = [Mob]
        spawn_obj._mob_args = Arguments(
            {
                "mob_location": None,
                "repeat_key": None,
                "repeat_location": None,
                "template_attr": template_attr,
            }
        )
        # the number of repetitions for Action if any
        spawn_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        spawn_obj.args = [Mob(**spawn_obj._mob_args)]

        # Repeat is optional, only add if default values were updated
        if spawn_obj._repeat_args.values_updated:
            spawn_obj.ARG_TYPES.append(Repeat)
            spawn_obj.args.append(Repeat(**spawn_obj._repeat_args))

        return spawn_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []

            for key in templ:
                # get the text form from template object.
                item = key.generate_description(arg_index=0, templ_index=j)

                if not item:
                    continue

                # Flatten if nested dict.
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Freebuild(ActionNode):
    """The Freebuild action uses the model to finish a block object that is half finished.
    The action takes a BlockObject.
    __init__(): Picks a template from freebuild_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of 'block_object_type',
                'block_object_attributes' etc for BlockObject
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Freebuild", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        freebuild_obj = Freebuild(template, template_attr)
        template = freebuild_obj.template

        freebuild_obj.ARG_TYPES = []
        freebuild_obj._no_children = False
        freebuild_obj._only_location = False  # If the object only has location
        freebuild_obj._block_obj_args = Arguments(
            {
                "block_object_type": Object,
                "block_object_attributes": [],
                "block_object_location": False,
                "no_child": False,
                "repeat_key": None,
                "repeat_no_child": None,
                "repeat_location": None,
                "coref_type": None,
                "template_attr": template_attr,
            }
        )
        freebuild_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "coref_resolve": None,
                "template_attr": template_attr,
            }
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        freebuild_obj.args = []

        if not freebuild_obj._only_location:
            freebuild_obj.ARG_TYPES.append(BlockObject)
            freebuild_obj.args.append(BlockObject(**freebuild_obj._block_obj_args))

        # Location is optional, only add if the default arguments were changed.
        if freebuild_obj._location_args.values_updated:
            freebuild_obj.ARG_TYPES.append(Location)
            freebuild_obj.args.append(Location(**freebuild_obj._location_args))

        return freebuild_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []

            for key in templ:
                arg_index = 0
                item = key.generate_description(arg_index=arg_index, templ_index=j)

                if not item:
                    continue
                # flatten if nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Dance(ActionNode):
    """The Dance action represents dancing/ moving in a defined way.
    The action takes an optional Location.
    __init__(): Picks a template from dance_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of 'location_type',
                'relative_direction' etc for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Dance", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        dance_obj = Dance(template, template_attr)
        template = dance_obj.template

        dance_obj.ARG_TYPES = []
        dance_obj._no_children = False  # no ARG_TYPE if _no_children is True
        dance_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "coref_resolve": None,
                "relative_direction_value": None,
                "template_attr": template_attr,
            }
        )

        dance_obj._condition_args = Arguments({"condition_type": None, "block_type": None})
        # the number of repetitions for Action if any
        dance_obj._repeat_args = Arguments(
            {"repeat_key": None, "repeat_count": None, "repeat_dir": None}
        )

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        dance_obj.args = []
        if dance_obj._no_children:
            return dance_obj

        if dance_obj._location_args.values_updated:
            dance_obj.ARG_TYPES.append(Location)
            dance_obj.args.append(Location(**dance_obj._location_args))

        # StopCondition is optional, only add if the default arguments were changed.
        if dance_obj._condition_args.values_updated:
            dance_obj.ARG_TYPES.append(StopCondition)
            dance_obj.args.append(StopCondition(**dance_obj._condition_args))

        # Repeat is optional, only add if default values were updated
        if dance_obj._repeat_args.values_updated:
            dance_obj.ARG_TYPES.append(Repeat)
            dance_obj.args.append(Repeat(**dance_obj._repeat_args))

        return dance_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []

            for i, key in enumerate(templ):
                key_type = type(key)
                arg_index = 0
                # arg_index 1 for StopCondition
                if (key_type in CONDIITON_TEMPLATES) and (len(self.args) > 1):
                    arg_index = 1

                # get the text from template object
                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Stop(ActionNode):
    """Stop action takes no arguments, and only has a description.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Stop", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        stop_obj = Stop(template, template_attr)
        template = stop_obj.template

        stop_obj.ARG_TYPES = []
        stop_obj._no_children = False  # no ARG_TYPE if _no_children is True
        stop_obj.target_action_type = None  # name of action to be undone
        stop_obj.args = []

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        return stop_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            arg_index = 0

            for i, key in enumerate(templ):
                item = None
                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Resume(ActionNode):
    """Resume action takes no arguments and only has a description.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("Resume", template, template_attr=template_attr)

    def generate(template=None, template_attr={}):
        resume_obj = Resume(template, template_attr)
        template = resume_obj.template

        resume_obj.ARG_TYPES = []
        resume_obj._no_children = False  # no ARG_TYPE if _no_children is True
        resume_obj.target_action_type = None  # name of action to be undone
        resume_obj.args = []

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        return resume_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            arg_index = 0

            for i, key in enumerate(templ):
                item = None
                item = key.generate_description(arg_index=arg_index, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class Noop(ActionNode):
    """Incoming messages which do not correspond to any action are mapped to Noop.
    """

    CHATS = ["hello there", "how are you", "great"]

    def __init__(self, template_attr={}):
        super().__init__("Noop", template_attr=template_attr)
        self._is_dialogue = True

    def _generate_description(self):
        self._word = random.choice(self.CHATS)
        return [self._word]  # ["human: " + self._word]
