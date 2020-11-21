"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Spawn templates are written for a MobName and represent the intent
for the action: Spawn.

Examples:
[Human, Spawn, MobName]
- spawn a pig.
- spawn sheep

[Human, Spawn, RepeatCount, MobName]
- Spawn five pigs
- Spawn a few sheep
    etc
'''


from template_objects import *


SPAWN_TEMPLATES = [
    ## Spawn mob ##
    [Human, Spawn, MobName],

    ## Spawn mob on the ground ##
    [Human, Spawn, MobName, OnGround],

    ## Spawn n mobs ##
    [Human, Spawn, RepeatCount, MobName],

    ## Spawn mob n times ##
    [Human, Spawn, MobName, NTimes]
]
