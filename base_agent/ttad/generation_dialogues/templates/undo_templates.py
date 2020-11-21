"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Undo templates are written for an action name and represents the intent
for the action: Undo. This action represents reverting something that has been
done before.

Examples:
[Human, Undo]
- undo last action
- undo what you just did

[Human, Undo, UndoActionBuild]
- undo what you just built
- undo the build action
'''


from template_objects import *


UNDO_TEMPLATES = [
    [Human, Undo],

    ## Undo action name ##
    [Human, Undo, ActionBuild],
    [Human, Undo, ActionDestroy],
    [Human, ActionTag],
    [Human, Undo, ActionFill],
    [Human, Undo, ActionDig]
]
