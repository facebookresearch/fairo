"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Dance templates are written with an optional location and stop condition.

Examples:
[Human, DanceSingle]
- do a dance
- dance

[Human, DanceSingle, ConditionTypeNever],
- keep dancing
- dance until I tell you to stop
'''

DANCE_WITH_CORRECTION = [
    [[Human, Dance],
     [HumanReplace, Dance, AroundString]],
    [[Human, Jump],
     [HumanReplace, Jump, AroundString]],
    [[Human, Fly],
     [HumanReplace, Fly, AroundString]],
    [[Human, Hop],
     [HumanReplace, Hop, AroundString]],
]

DANCE_TEMPLATES = [
    ## Dance single word ##
    [Human, Dance],
    [Human, Dance, ConditionTypeNever],

    [Human, Fly],
    [Human, Fly, ConditionTypeNever],

    [Human, Jump],
    [Human, Jump, ConditionTypeNever],

    [Human, Hop],
    [Human, Hop, ConditionTypeNever],

    ## Walk around X ##
    [Human, Fly, AroundString, LocationBlockObjectTemplate],
    [Human, Jump, AroundString, LocationBlockObjectTemplate],
    [Human, Hop, AroundString, LocationBlockObjectTemplate],
    [Human, Dance, AroundString, LocationBlockObjectTemplate],
    [Human, Walk, AroundString, LocationBlockObjectTemplate],

    ## Move around X clockwise / anticlockwise ##
    [Human, Fly, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate],
    [Human, Jump, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate],
    [Human, Hop, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate],
    [Human, Dance, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate],
    [Human, Walk, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate],

    ## move around X clockwise/anticlockwise n times ##
    [Human, Dance, NTimes],
    [Human, Fly, NTimes],
    [Human, Jump, NTimes],
    [Human, Hop, NTimes],

    [Human, Fly, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate, NTimes],
    [Human, Jump, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate, NTimes],
    [Human, Hop, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate, NTimes],
    [Human, Dance, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate, NTimes],
    [Human, Walk, AroundString, LocationBlockObjectTemplate, RelativeDirectionTemplate, NTimes],
]
