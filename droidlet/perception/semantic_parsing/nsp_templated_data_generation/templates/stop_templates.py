"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Stop templates are written for an action name and represents the intent
for the action: Stop. This action represents stopping an action

Examples:
[Human, StopSingle]
- Stop
- pause

[Human, Stop, ActionBuild]
- Stop building
'''

STOP_TEMPLATES = [
    [Human, StopSingle],

    ## Stop action name ##
    [Human, Stop, ActionBuild],
    [Human, Stop, ActionDestroy],
    [Human, Stop, ActionTag],
    [Human, Stop, ActionFill],
    [Human, Stop, ActionDig],
    [Human, Stop, ActionMove],

    ## Dont do action ##
    [Human, Dont, ActionBuild],
    [Human, Dont, ActionDestroy],
    [Human, Dont, ActionTag],
    [Human, Dont, ActionFill],
    [Human, Dont, ActionDig],
    [Human, Dont, ActionMove],
]
