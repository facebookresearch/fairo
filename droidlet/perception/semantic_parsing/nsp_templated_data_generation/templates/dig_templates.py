"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Dig templates are written for a Location and represent the intent
for the action: Dig. This action intends to dig a hole at a certain location.

Examples:
[Human, Dig, DigSomeShape, ThereTemplate]
- dig a hole there
- make a tunnel there

[Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate]
- dig a hole a little to the right of the sheep
- make a tunnel a bit in front of the pig
'''

DIG_WITH_CORRECTION = [
    ## General pattern : dig + new location specification
    ## Dig X N times, add location ##
    [[Human, Dig, DigSomeShape, NTimes],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigDimensions, DigSomeShape, NTimes],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigAbstractSize, DigSomeShape, NTimes],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigDimensions, NTimes],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigDimensions, DigSomeShape, NTimes],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, DigSomeShape],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, ThereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, HereTemplateCoref],
    [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, ThereTemplateCoref, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, HereTemplate, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, HereTemplateCoref, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, YouTemplate, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, DownTo, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, DownTo, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigSomeShape, DownTo, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, DigSomeShape, ThereTemplate, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, ThereTemplateCoref, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, HereTemplate, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, HereTemplateCoref, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, YouTemplate, ConditionTypeAdjacentBlockType],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    ## Dig an X by Y ##
    [[Human, Dig, DigDimensions],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    ## Dig X of dimensions Y ##
    [[Human, Dig, DigSomeShape, OfDimensionsPhrase, DigDimensions],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, DigDimensions, DigSomeShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigDimensions, DigSomeShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigDimensions, DigSomeShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigDimensions, DigSomeShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, DigAbstractSize, DigSomeShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigAbstractSize, DigSomeShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigAbstractSize, DigSomeShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigAbstractSize, DigSomeShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    ## Dig X Y blocks long / wide / deep ##
    [[Human, Dig, DigSomeShape, NumBlocks, Squares, Wide],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, NumBlocks, Squares, Long],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, DigSomeShape, NumBlocks, Squares, Deep],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, RepeatCount, DigSomeShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigSomeShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigSomeShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigSomeShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigDimensions, DigSomeShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigDimensions, DigSomeShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigDimensions, DigSomeShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ThereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, HereTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, HereTemplateCoref],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, YouTemplate],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],

    [[Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
    [[Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep],
     [HumanReplace, RelativeDirectionTemplate, LocationMobTemplate, Please]],
]
DIG_TEMPLATES = [
    ## Single word Dig command ##
    [Human, DigSingle],

    ## Dig at location X (optional) ##
    [Human, Dig, Under],
    [Human, Dig, Under, YouTemplate],
    [Human, Dig, YouTemplate],
    [Human, Dig, HereTemplate],
    [Human, Dig, HereTemplateCoref],
    [Human, Dig, ThereTemplate],
    [Human, Dig, ThereTemplateCoref],
    [Human, Dig, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Dig, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Dig X N times ##
    [Human, Dig, DigSomeShape, NTimes],
    [Human, Dig, DigDimensions, DigSomeShape, NTimes],
    [Human, Dig, DigAbstractSize, DigSomeShape, NTimes],
    [Human, Dig, DigDimensions, NTimes],
    [Human, Dig, DigDimensions, DigSomeShape, NTimes],

    ## Dig X at location Y (optional) ##
    [Human, Dig, DigSomeShape],
    [Human, Dig, DigSomeShape, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Dig, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Dig, DigSomeShape, ThereTemplate],
    [Human, Dig, DigSomeShape, ThereTemplateCoref],
    [Human, Dig, DigSomeShape, HereTemplate],
    [Human, Dig, DigSomeShape, HereTemplateCoref],
    [Human, Dig, DigSomeShape, YouTemplate],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Dig at location X (optional) until condition Y ##
    [Human, Dig, ConditionTypeAdjacentBlockType],
    [Human, Dig, At, LocationWord, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, ConditionTypeAdjacentBlockType],

    [Human, Dig, ThereTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, ThereTemplateCoref, ConditionTypeAdjacentBlockType],
    [Human, Dig, HereTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, HereTemplateCoref, ConditionTypeAdjacentBlockType],
    [Human, Dig, YouTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, ALittle, RelativeDirectionTemplate, YouTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, ALittle, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],

    ## Dig down to block type X ##
    [Human, Dig, DownTo, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, DownTo, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, DownTo, ConditionTypeAdjacentBlockType],

    ## Dig X at location Y (optional) until condition Z ##
    [Human, Dig, DigSomeShape, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, At, LocationWord, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, ConditionTypeAdjacentBlockType],

    [Human, Dig, DigSomeShape, ThereTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, ThereTemplateCoref, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, HereTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, HereTemplateCoref, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, YouTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],

    ## Dig an X by Y ##
    [Human, Dig, DigDimensions],

    ## Dig X of dimensions Y ##
    [Human, Dig, DigSomeShape, OfDimensionsPhrase, DigDimensions],

    ## Dig a dimension X shape Y at location Z (optional) ##
    [Human, Dig, DigDimensions, DigSomeShape],
    [Human, Dig, DigDimensions, DigSomeShape, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Dig, DigDimensions, DigSomeShape, ThereTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, ThereTemplateCoref],
    [Human, Dig, DigDimensions, DigSomeShape, HereTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, HereTemplateCoref],
    [Human, Dig, DigDimensions, DigSomeShape, YouTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Dig a size X shape Y at location Z (optional) ##
    [Human, Dig, DigAbstractSize, DigSomeShape],
    [Human, Dig, DigAbstractSize, DigSomeShape, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, DigAbstractSize, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ThereTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ThereTemplateCoref],
    [Human, Dig, DigAbstractSize, DigSomeShape, HereTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, HereTemplateCoref],
    [Human, Dig, DigAbstractSize, DigSomeShape, YouTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Dig X Y blocks long / wide / deep ##
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep],

    ## Dig X Y blocks long and Z blocks deep ##
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long],

    ## Dig X Y blocks long Z blocks deep ##
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Long],

    ## Dig X Y blocks long and Z blocks deep and Z blocks wide ##
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep],

    ## Dig X Y blocks long Z blocks deep Z and blocks wide ##
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep],

    ## Dig X Y blocks long Z blocks deep Z blocks wide ##
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Long, NumBlocks, Squares, Deep],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide, NumBlocks, Squares, Long],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Long, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide],
    [Human, Dig, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep],

    ##Dig at every location X ##
    [Human, Dig, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig X at every location Y ##
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig dimension X shape Y at every location Z ##
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig size X shape Y at every location Z ##
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, RepeatAllLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig N Holes at location Y (optional) ##
    [Human, Dig, RepeatCount, DigSomeShape],
    [Human, Dig, RepeatCount, DigSomeShape, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, ThereTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, ThereTemplateCoref],
    [Human, Dig, RepeatCount, DigSomeShape, HereTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, HereTemplateCoref],
    [Human, Dig, RepeatCount, DigSomeShape, YouTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Dig N holes at location Y until condition Z ##
    [Human, Dig, RepeatCount, DigSomeShape, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, At, LocationWord, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, ConditionTypeAdjacentBlockType],

    [Human, Dig, RepeatCount, DigSomeShape, ThereTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, ThereTemplateCoref, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, HereTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, HereTemplateCoref, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, YouTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, ConditionTypeAdjacentBlockType],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, ConditionTypeAdjacentBlockType],

    ## Dig N holes of dimension Y at location Z ##
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape],
    [Human, Dig, RepeatCount, DigSomeShape, OfDimensionsPhrase, DigDimensions],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ThereTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ThereTemplateCoref],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, HereTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, HereTemplateCoref],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, YouTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Dig N holes of size Y at location Z ##
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, At, LocationWord, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ThereTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ThereTemplateCoref],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, HereTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, HereTemplateCoref],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, YouTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Dig N holes X blocks wide ##
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep],

    ## Dig N holes X blocks wide and Y blocks long ##
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long],

    ## Dig N holes X blocks wide Y blocks long ##
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Long],

    ## Dig N holes X blocks wide and Y blocks long and Z blocks deep ##
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep],

    ## Dig N holes X blocks wide Y blocks long and Z blocks deep ##
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Long, And, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Long, And, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Deep, And, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Wide, And, NumBlocks, Squares, Deep],

    ## Dig N holes X blocks wide Y blocks long Z blocks deep ##
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Wide, NumBlocks, Squares, Long, NumBlocks, Squares, Deep],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide, NumBlocks, Squares, Long],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Deep, NumBlocks, Squares, Long, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Deep, NumBlocks, Squares, Wide],
    [Human, Dig, RepeatCount, DigSomeShape, NumBlocks, Squares, Long, NumBlocks, Squares, Wide, NumBlocks, Squares, Deep],

    ## Dig N X at every location Y ###
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig N dimension X Y at every location Z ##
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig N size X Y at every location Z ##
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, RepeatAllLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Dig at locations X ##
    [Human, Dig, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Dig X at locations Y ##
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Dig dimension X Y at locations Z ##
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Dig size X Y at locations Z ##
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, RepeatCountLocation],
    [Human, Dig, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Dig N X at locations Y ##
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Dig N dimension X Y at locations Z ##
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigDimensions, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Dig N size X Y at locations Z ##
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, CoordinatesTemplate, RepeatCountLocation],
    [Human, Dig, RepeatCount, DigAbstractSize, DigSomeShape, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    ]
