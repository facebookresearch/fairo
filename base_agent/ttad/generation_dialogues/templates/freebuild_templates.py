"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Freebuild templates are written for :
- either a BlockObject or a Location.
and represents the intent for the action: Freebuild.

This action intends for a generation model to complete or finish something that
is half-built.

Examples:
[Human, Freebuild, The, Colour, AbstractDescription, BlockObjectLocation]
- complete the red structure to the left of that grey shape.
- finish the blue thing at location: 2 , 3, 4 for me

[Human, Freebuild, BlockObjectThat, Size, Colour, AbstractDescription]
- complete that huge red structure
- finish that tiny blue thing for me please
    etc
'''


from base_agent.ttad.generation_dialogues.template_objects import *


FREEBUILD_WITH_CORRECTION = [
    [[Human, FreebuildLocation, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, FreebuildLocation, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, FreebuildLocation, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, FreebuildLocation, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate]],

    [[Human, FreebuildLocation, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, FreebuildLocation, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, FreebuildLocation, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, FreebuildLocation, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, BlockObjectThat]],

    [[Human, Freebuild, BlockObjectThat, ForMe],
     [HumanReplace, The, Colour, AbstractDescription]],
    [[Human, Freebuild, BlockObjectThat, ForMe],
     [HumanReplace, The, Size, AbstractDescription]],
    [[Human, Freebuild, BlockObjectThat, ForMe],
     [HumanReplace, The, Size, Colour, AbstractDescription]],

    [[Human, Freebuild, BlockObjectThis, ForMe],
     [HumanReplace, The, Colour, AbstractDescription]],
    [[Human, Freebuild, BlockObjectThis, ForMe],
     [HumanReplace, The, Size, AbstractDescription]],
    [[Human, Freebuild, BlockObjectThis, ForMe],
     [HumanReplace, The, Size, Colour, AbstractDescription]],

    [[Human, Freebuild, BlockObjectIt, ForMe],
     [HumanReplace, The, Colour, AbstractDescription]],
    [[Human, Freebuild, BlockObjectIt, ForMe],
     [HumanReplace, The, Size, AbstractDescription]],
    [[Human, Freebuild, BlockObjectIt, ForMe],
     [HumanReplace, The, Size, Colour, AbstractDescription]],

    [[Human, Freebuild, BlockObjectThat, Colour, AbstractDescription, ForMe],
     [HumanReplace, The, Size, One]],
    [[Human, Freebuild, BlockObjectThis, Colour, AbstractDescription, ForMe],
     [HumanReplace, The, Size, One]],
    [[Human, Freebuild, The, Colour, AbstractDescription, ForMe],
     [HumanReplace, The, Size, One]],
    [[Human, Freebuild, BlockObjectThat, Colour, ConcreteDescription, ForMe],
     [HumanReplace, The, Size, One]],
    [[Human, Freebuild, BlockObjectThis, Colour, ConcreteDescription, ForMe],
     [HumanReplace, The, Size, One]],
    [[Human, Freebuild, The, Colour, ConcreteDescription, ForMe],
     [HumanReplace, The, Size, One]],

    [[Human, Freebuild, BlockObjectThat, AbstractDescription, ForMe],
     [HumanReplace, The, Size, One]],
    [[Human, Freebuild, BlockObjectThat, AbstractDescription, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, BlockObjectThat, AbstractDescription, ForMe],
     [HumanReplace, The, Size, Colour, One]],

     [[Human, Freebuild, BlockObjectThis, AbstractDescription, ForMe],
      [HumanReplace, The, Size, One]],
     [[Human, Freebuild, BlockObjectThis, AbstractDescription, ForMe],
      [HumanReplace, The, Colour, One]],
     [[Human, Freebuild, BlockObjectThis, AbstractDescription, ForMe],
      [HumanReplace, The, Size, Colour, One]],

     [[Human, Freebuild, BlockObjectThat, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, One]],
     [[Human, Freebuild, BlockObjectThat, ConcreteDescription, ForMe],
      [HumanReplace, The, Colour, One]],
     [[Human, Freebuild, BlockObjectThat, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, Colour, One]],

     [[Human, Freebuild, BlockObjectThis, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, One]],
     [[Human, Freebuild, BlockObjectThis, ConcreteDescription, ForMe],
      [HumanReplace, The, Colour, One]],
     [[Human, Freebuild, BlockObjectThis, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, Colour, One]],

    [[Human, Freebuild, BlockObjectThat, Size, AbstractDescription, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, BlockObjectThis, Size, AbstractDescription, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, The, Size, AbstractDescription, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, BlockObjectThat, Size, ConcreteDescription, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, BlockObjectThis, Size, ConcreteDescription, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, The, Size, ConcreteDescription, ForMe],
     [HumanReplace, The, Colour, One]],

    [[Human, Freebuild, The, Size, AbstractDescription, BlockObjectLocation, ForMe],
     [HumanReplace, The, Colour, One]],
    [[Human, Freebuild, The, Size, ConcreteDescription, BlockObjectLocation, ForMe],
     [HumanReplace, The, Colour, One]],

     [[Human, Freebuild, RepeatCount, ConcreteDescription, ForMe],
      [HumanReplace, The, Colour, One]],
     [[Human, Freebuild, RepeatCount, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, One]],
     [[Human, Freebuild, RepeatCount, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, Colour, One]],
     [[Human, Freebuild, RepeatCount, Colour, ConcreteDescription, ForMe],
      [HumanReplace, The, Size, One]],

     [[Human, Freebuild, RepeatCount, Size, AbstractDescription, ForMe],
      [HumanReplace, The, Colour, One]],
     [[Human, Freebuild, RepeatCount, Size, ConcreteDescription, ForMe],
      [HumanReplace, The, Colour, One]],

     [[Human, Freebuild, RepeatCount, Size, AbstractDescription, BlockObjectLocation, ForMe],
      [HumanReplace, The, Colour, One]],
     [[Human, Freebuild, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, ForMe],
      [HumanReplace, The, Colour, One]],

     [[Human, Freebuild, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, ForMe],
      [HumanReplace, The, Size, One]],
     [[Human, Freebuild, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, ForMe],
      [HumanReplace, The, Size, One]],
    ]

FREEBUILD_TEMPLATES = [
    ## Freebuild with only Location ##
    [Human, FreebuildLocation, At, LocationWord, CoordinatesTemplate],
    [Human, FreebuildLocation, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, FreebuildLocation, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, FreebuildLocation, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, FreebuildLocation, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, FreebuildLocation, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, FreebuildLocation, RelativeDirectionTemplate, BlockObjectThat],
    [Human, FreebuildLocation, ThereTemplate],
    [Human, FreebuildLocation, ThereTemplateCoref],
    [Human, FreebuildLocation, HereTemplate],
    [Human, FreebuildLocation, HereTemplateCoref],
    [Human, FreebuildLocation, YouTemplate],
    [Human, FreebuildLocation, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, FreebuildLocation, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, FreebuildLocation, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, FreebuildLocation, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, FreebuildLocation, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, FreebuildLocation, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, FreebuildLocation, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, FreebuildLocation, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, FreebuildLocation, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Freebuild with only BlockObject ##
    [Human, Freebuild, BlockObjectThat, ForMe],
    [Human, Freebuild, BlockObjectThis, ForMe],
    [Human, Freebuild, BlockObjectIt, ForMe],

    [Human, Freebuild, BlockObjectThat, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThat, ConcreteDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, ConcreteDescription, ForMe],

    [Human, Freebuild, The, ConcreteDescription, ForMe],
    [Human, Freebuild, ConcreteDescription, ForMe],

    [Human, Freebuild, The, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, The, ConcreteDescription, BlockObjectLocation, ForMe],

    [Human, Freebuild, BlockObjectThat, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, The, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThat, Colour, ConcreteDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, Colour, ConcreteDescription, ForMe],
    [Human, Freebuild, The, Colour, ConcreteDescription, ForMe],

    [Human, Freebuild, BlockObjectThat, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThat, ConcreteDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, ConcreteDescription, ForMe],

    [Human, Freebuild, BlockObjectThat, Size, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, Size, AbstractDescription, ForMe],
    [Human, Freebuild, The, Size, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThat, Size, ConcreteDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, Size, ConcreteDescription, ForMe],
    [Human, Freebuild, The, Size, ConcreteDescription, ForMe],

    [Human, Freebuild, The, Size, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, The, Size, ConcreteDescription, BlockObjectLocation, ForMe],

    [Human, Freebuild, The, Colour, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, The, Colour, ConcreteDescription, BlockObjectLocation, ForMe],

    [Human, Freebuild, BlockObjectThat, Size, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, Size, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, The, Size, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, BlockObjectThat, Size, Colour, ConcreteDescription, ForMe],
    [Human, Freebuild, BlockObjectThis, Size, Colour, ConcreteDescription, ForMe],
    [Human, Freebuild, The, Size, Colour, ConcreteDescription, ForMe],

    [Human, Freebuild, The, Size, Colour, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, The, Size, Colour, ConcreteDescription, BlockObjectLocation, ForMe],

    ### Freebuild num X ###
    [Human, Freebuild, RepeatCount, ConcreteDescription, ForMe],
    [Human, Freebuild, RepeatCount, Colour, ConcreteDescription, ForMe],

    [Human, Freebuild, RepeatCount, Size, AbstractDescription, ForMe],
    [Human, Freebuild, RepeatCount, Size, ConcreteDescription, ForMe],

    [Human, Freebuild, RepeatCount, Size, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, ForMe],

    [Human, Freebuild, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, ForMe],

    [Human, Freebuild, RepeatCount, Size, Colour, AbstractDescription, ForMe],
    [Human, Freebuild, RepeatCount, Size, Colour, ConcreteDescription, ForMe],

    [Human, Freebuild, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, ForMe],
    [Human, Freebuild, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, ForMe],

    ### Freebuild X in front of num Y ###
    [Human, Freebuild, The, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, The, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    [Human, Freebuild, The, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, The, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    [Human, Freebuild, The, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, The, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    ### Freebuild num X in front of num Y ###
    [Human, Freebuild, RepeatCount, AbstractDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, RepeatCount, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    [Human, Freebuild, RepeatCount, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    [Human, Freebuild, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    [Human, Freebuild, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, ForMe],
    [Human, Freebuild, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, ForMe],

    ### Freebuild All X in front of Y ###
    [Human, Freebuild, All, ConcreteDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, ConcreteDescription, RepeatAll, ForMe],

    [Human, Freebuild, All, Colour, ConcreteDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, ConcreteDescription, RepeatAll, ForMe],
    [Human, Freebuild, All, Colour, AbstractDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, AbstractDescription, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, AbstractDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, AbstractDescription, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, ConcreteDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, ConcreteDescription, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, AbstractDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, ConcreteDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, AbstractDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, ConcreteDescription, BlockObjectLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Colour, AbstractDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, AbstractDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, Colour, AbstractDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, AbstractDescription, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, Colour, ConcreteDescription, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, ConcreteDescription, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll, ForMe],

    ### Freebuild X in front of all Ys ###
    [Human, Freebuild, The, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    [Human, Freebuild, The, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    [Human, Freebuild, The, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    [Human, Freebuild, The, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    [Human, Freebuild, The, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    [Human, Freebuild, The, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    [Human, Freebuild, The, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    ### Freebuild all X in front of all Ys ###
    [Human, Freebuild, All, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll, ForMe],

    ### Freebuild all X in front of n Ys ###
    [Human, Freebuild, All, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],

    [Human, Freebuild, All, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, All, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],
    [Human, Freebuild, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll, ForMe],

    ### Freebuild n X in front of all Ys ###
    [Human, Freebuild, RepeatCount, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    [Human, Freebuild, RepeatCount, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    [Human, Freebuild, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    [Human, Freebuild, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    [Human, Freebuild, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],

    [Human, Freebuild, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    [Human, Freebuild, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, ForMe],
    ] 
