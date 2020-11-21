"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Resume templates are written for an action name and represents the intent
for the action: Resume. This action represents resuming a given or last action.

Examples:
[Human, ResumeSingle]
- resume
- start again

[Human, Resume, ActionBuild]
- resume building
- continue the build action
'''


from template_objects import *


RESUME_TEMPLATES = [
    [Human, ResumeSingle],

    ## Resume action name ##
    [Human, Resume, ActionBuild],
    [Human, Resume, ActionDestroy],
    [Human, Resume, ActionTag],
    [Human, Resume, ActionFill],
    [Human, Resume, ActionDig],
    [Human, Resume, ActionMove],

]
