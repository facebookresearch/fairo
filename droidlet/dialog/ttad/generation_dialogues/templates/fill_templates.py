"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Fill templates are written for a Location and represents the intent
for the action: Fill. This action intends to fill a hole or a negative shape
at a certain location.

Examples:
[Human, Fill, The, FillShape]
- fill the hole
- cover up the mine

[Human, Fill, The, FillShape, At, LocationWord, CoordinatesTemplate]
- fill up the hole at location: 2, 3, 4
- cover up the tunnel at: 2, 3, 4
'''


from droidlet.dialog.ttad.generation_dialogues.template_objects import *

FILL_WITH_CORRECTION = [
    ## Single word command ##
    [[Human, Fill],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Fill],
     [HumanReplace, UseFill, FillBlockType]],

    ## Fill up the shape ##
    [[Human, Fill, The, FillShape],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, The, FillShape, Up],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Fill, The, FillShape],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, The, FillShape, Up],
     [HumanReplace, UseFill, FillBlockType]],

    ## Fill shape X at location Y ##
    [[Human, Fill, FillObjectThis, FillShape],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Fill, The, FillShape, ThereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, The, FillShape, HereTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, The, FillShape, HereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, The, FillShape, YouTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Fill, FillObjectThis, RepeatCount, FillShape],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatCount, FillShape, ThereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatCount, FillShape, HereTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatCount, FillShape, HereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatCount, FillShape, YouTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Fill, RepeatAll, FillShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatAll, FillShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatAll, FillShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Fill, RepeatAll, FillShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    ## Fill with X ##
    [[Human, Fill, FillObjectThis, FillShape],
     [HumanReplace, UseFill, FillBlockType]],

    [[Human, Fill, The, FillShape, ThereTemplateCoref],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, The, FillShape, HereTemplate],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, The, FillShape, HereTemplateCoref],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, The, FillShape, YouTemplate],
     [HumanReplace, UseFill, FillBlockType]],

    [[Human, Fill, FillObjectThis, RepeatCount, FillShape],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatCount, FillShape, ThereTemplateCoref],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatCount, FillShape, HereTemplate],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatCount, FillShape, HereTemplateCoref],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatCount, FillShape, YouTemplate],
     [HumanReplace, UseFill, FillBlockType]],

    [[Human, Fill, RepeatAll, FillShape, ThereTemplateCoref],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatAll, FillShape, HereTemplate],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatAll, FillShape, HereTemplateCoref],
     [HumanReplace, UseFill, FillBlockType]],
    [[Human, Fill, RepeatAll, FillShape, YouTemplate],
     [HumanReplace, UseFill, FillBlockType]],

    ## All rel_dir to BlockObject templates ##
     ## Single word command ##
    [[Human, Fill],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat, Please]],

    ## Fill up the shape ##
    [[Human, Fill, The, FillShape],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, The, FillShape, Up],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],

    ## Fill shape X at location Y ##
    [[Human, Fill, FillObjectThis, FillShape],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],

    [[Human, Fill, The, FillShape, ThereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, The, FillShape, HereTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, The, FillShape, HereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, The, FillShape, YouTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],

    [[Human, Fill, FillObjectThis, RepeatCount, FillShape],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatCount, FillShape, ThereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatCount, FillShape, HereTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatCount, FillShape, HereTemplateCoref],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatCount, FillShape, YouTemplate],
     [HumanReplace, The, One, RelativeDirectionTemplate, BlockObjectThat, Please]],

    [[Human, Fill, RepeatAll, FillShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatAll, FillShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatAll, FillShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat, Please]],
    [[Human, Fill, RepeatAll, FillShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat, Please]],
]

FILL_TEMPLATES = [
    ## Single word command ##
    [Human, Fill],
    [Human, Fill, Using, FillBlockType],

    ## Fill up the shape ##
    [Human, Fill, The, FillShape],
    [Human, Fill, The, FillShape, Up],
    [Human, Fill, The, FillShape, Using, FillBlockType],
    [Human, Fill, The, FillShape, Up, Using, FillBlockType],

    ## Fill shape X at location Y ##
    [Human, Fill, FillObjectThis, FillShape],
    [Human, Fill, The, FillShape, At, LocationWord, CoordinatesTemplate],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, The, FillShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Fill, The, FillShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Fill, The, FillShape, ThereTemplate],
    [Human, Fill, The, FillShape, ThereTemplateCoref],
    [Human, Fill, The, FillShape, HereTemplate],
    [Human, Fill, The, FillShape, HereTemplateCoref],
    [Human, Fill, The, FillShape, YouTemplate],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Fill, FillObjectThis, FillShape, Using, FillBlockType],
    [Human, Fill, The, FillShape, At, LocationWord, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, ThereTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, ThereTemplateCoref, Using, FillBlockType],
    [Human, Fill, The, FillShape, HereTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, HereTemplateCoref, Using, FillBlockType],
    [Human, Fill, The, FillShape, YouTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, YouTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],

    ## Fill n holes ##
    [Human, Fill, FillObjectThis, RepeatCount, FillShape],
    [Human, Fill, RepeatCount, FillShape, At, LocationWord, CoordinatesTemplate],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, RepeatCount, FillShape, ThereTemplate],
    [Human, Fill, RepeatCount, FillShape, ThereTemplateCoref],
    [Human, Fill, RepeatCount, FillShape, HereTemplate],
    [Human, Fill, RepeatCount, FillShape, HereTemplateCoref],
    [Human, Fill, RepeatCount, FillShape, YouTemplate],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Fill, FillObjectThis, RepeatCount, FillShape, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, At, LocationWord, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ThereTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ThereTemplateCoref, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, HereTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, HereTemplateCoref, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, YouTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, YouTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],

    ## Fill a hole near every location X ##
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation, Using, FillBlockType],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation, Using, FillBlockType],

    ## Fill a hole near n locations X ##
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation, Using, FillBlockType],
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation, Using, FillBlockType],

    ## Fill all holes ##
    [Human, Fill, RepeatAll, FillShape, At, LocationWord, CoordinatesTemplate],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, RepeatAll, FillShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Fill, RepeatAll, FillShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Fill, RepeatAll, FillShape, ThereTemplate],
    [Human, Fill, RepeatAll, FillShape],
    [Human, Fill, RepeatAll, FillShape, ThereTemplateCoref],
    [Human, Fill, RepeatAll, FillShape, HereTemplate],
    [Human, Fill, RepeatAll, FillShape, HereTemplateCoref],
    [Human, Fill, RepeatAll, FillShape, YouTemplate],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Fill, RepeatAll, FillShape, At, LocationWord, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ThereTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ThereTemplateCoref, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, HereTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, HereTemplateCoref, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, YouTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, YouTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, Using, FillBlockType],

    ## Fill n holes near m locations X ##
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation, Using, FillBlockType],

    ## Fill all holes near n locations X ##
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation, Using, FillBlockType],

    ## Fill n holes near every location X ##
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation, Using, FillBlockType],

    # ## All rel_dir to BlockObjectThat templates ##
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Fill, The, FillShape, Between, BlockObjectThese],
    [Human, Fill, The, FillShape, Between, BlockObjectThose],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Fill, The, FillShape, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],
    [Human, Fill, The, FillShape, Between, BlockObjectThese, Using, FillBlockType],
    [Human, Fill, The, FillShape, Between, BlockObjectThose, Using, FillBlockType],
    [Human, Fill, The, FillShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],
    [Human, Fill, The, FillShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],

    ## Fill n holes ##
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],
    [Human, Fill, RepeatCount, FillShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],

    ## Fill a hole near every location X ##
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Fill, The, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation, Using, FillBlockType],

    ## Fill a hole near n locations X ##
    [Human, Fill, The, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Fill, The, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation, Using, FillBlockType],

    ## Fill all holes ##
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Fill, RepeatAll, FillShape, Between, BlockObjectThese],
    [Human, Fill, RepeatAll, FillShape, Between, BlockObjectThose],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, Between, BlockObjectThese, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, Between, BlockObjectThose, Using, FillBlockType],
    [Human, Fill, RepeatAll, FillShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, Using, FillBlockType],

    ## Fill n holes near m locations X ##
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation, Using, FillBlockType],

    ## Fill all holes near n locations X ##
    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Fill, RepeatAll, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation, Using, FillBlockType],

     ## Fill n holes near every location X ##
    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Fill, RepeatCount, FillShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation, Using, FillBlockType]

] 
