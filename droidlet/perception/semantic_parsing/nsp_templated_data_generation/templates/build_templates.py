"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
"""
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Build templates are written for Schematics and may have a Location,
and represent the intent for the action: Build.
This action builds a known Schematic (a shape or CategoryObject).

Examples:

[Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate]
- out of acacia build a cube at location 3, 2, 1
- out of acacia wood construct a cube of size 5 at location 5 6 7
- make a cube of size 10 at location ( 1 , 2, 3 )

[Human, Near, LocationMobTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType]
- near the spider build a dome of radius 4 using gravel.
- near the spider make a dome.
- near the spider assemble a dome using gravel.
"""

BUILD_INBUILT_COMPOSITE = [
[Human, Build, DescribingWord, AndBuild],
]
BUILD_WITH_CORRECTION = [
    ## Single word Build command ##
    [[Human, BuildSingle],
     [HumanReplace, Use, UsingBlockType]],

    ## Build at location X ##
    [[Human, BuildSingle, At, LocationWord, CoordinatesTemplate],
     [HumanReplace, Use, UsingBlockType]],

    [[Human, BuildSingle, RelativeDirectionTemplate, CoordinatesTemplate],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, RelativeDirectionTemplate, LocationMobTemplate],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, ThereTemplate],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, ThereTemplateCoref],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, HereTemplate],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, HereTemplateCoref],
    [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, YouTemplate],
     [HumanReplace, Use, UsingBlockType]],

    ## Build using block type X at location Y ##
    [[Human, BuildSingle, MadeOutOf, UsingBlockType],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, ThereTemplate],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, ThereTemplateCoref],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, HereTemplate],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, HereTemplateCoref],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, YouTemplate],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],

    ## Build at every location Y ##
    [[Human, BuildSingle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
     [HumanReplace, Use, UsingBlockType]],

    ## Build at n locations Y ##
    [[Human, BuildSingle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, BuildSingle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
     [HumanReplace, Use, UsingBlockType]],

    ## Stack N blocks at location Y ##
    [[Human, Stack, RepeatCount, DescribingWord],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, Stack, RepeatCount, DescribingWord],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    [[Human, Place, RepeatCount, DescribingWord, InARow],
     [HumanReplace, Use, UsingBlockType]],
    [[Human, Place, RepeatCount, DescribingWord, InARow],
     [HumanReplace, Build, RelativeDirectionTemplate, LocationMobTemplate]],
    ## templates for rel_dir of BlockObjectThat ##
    [[Human, BuildSingle, RelativeDirectionTemplate, BlockObjectThat],
     [HumanReplace, Use, UsingBlockType]],

    ## Build using block type X at location Y ##
    [[Human, BuildSingle, MadeOutOf, UsingBlockType],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, ThereTemplate],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, ThereTemplateCoref],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, HereTemplate],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, HereTemplateCoref],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, BuildSingle, MadeOutOf, UsingBlockType, YouTemplate],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],

    ## Build at every location Y ##
    [[Human, BuildSingle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
     [HumanReplace, Use, UsingBlockType]],

    ## Build at n locations Y ##
    [[Human, BuildSingle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
     [HumanReplace, Use, UsingBlockType]],

    ## Stack N blocks at location Y ##
    [[Human, Stack, RepeatCount, DescribingWord],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
    [[Human, Place, RepeatCount, DescribingWord, InARow],
     [HumanReplace, Build, RelativeDirectionTemplate, BlockObjectThat]],
]

BUILD_TEMPLATES = [
    ## Single word Build command ##
    [Human, BuildSingle],

    ## Build at location X ##
    [Human, BuildSingle, At, LocationWord, CoordinatesTemplate],
    [Human, BuildSingle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, BuildSingle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, BuildSingle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, BuildSingle, ThereTemplate],
    [Human, BuildSingle, ThereTemplateCoref],
    [Human, BuildSingle, HereTemplateCoref],
    [Human, BuildSingle, HereTemplate],
    [Human, BuildSingle, YouTemplate],

    ## Build using block type X at location Y ##
    [Human, BuildSingle, MadeOutOf, UsingBlockType],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, ThereTemplateCoref],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, YouTemplate],

    ## Build at every location Y ##
    [Human, BuildSingle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, BuildSingle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build at n locations Y ##
    [Human, BuildSingle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, BuildSingle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    ## Single word shape name for Build ##
    [Human, BuildShape],

    ## Surround X with Y ##
    [Human, Surround, LocationBlockObjectTemplate, SurroundWith, DescribingWord],
    [Human, Surround, LocationMobTemplate, SurroundWith, DescribingWord],
    [Human, Surround, BlockObjectThat, SurroundWith, DescribingWord],

    ## Stack N blocks at location Y ##
    [Human, Stack, RepeatCount, DescribingWord],
    [Human, Stack, RepeatCount, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Stack, RepeatCount, DescribingWord, ThereTemplate],
    [Human, Stack, RepeatCount, DescribingWord, ThereTemplateCoref],
    [Human, Stack, RepeatCount, DescribingWord, HereTemplateCoref],
    [Human, Stack, RepeatCount, DescribingWord, HereTemplate],
    [Human, Stack, RepeatCount, DescribingWord, YouTemplate],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Stack N blocks at every location Y ##
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Stack N blocks at M locations Y ##
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    ## Place N blocks in a row at location Y ##
    [Human, Place, RepeatCount, DescribingWord, InARow],
    [Human, Place, RepeatCount, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, ThereTemplate],
    [Human, Place, RepeatCount, DescribingWord, ThereTemplateCoref],
    [Human, Place, RepeatCount, DescribingWord, HereTemplateCoref],
    [Human, Place, RepeatCount, DescribingWord, HereTemplate],
    [Human, Place, RepeatCount, DescribingWord, YouTemplate],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Place N blocks at every location Y ##
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Place N blocks in a row at location Y ##
    [Human, Place, RepeatCount, DescribingWord, InARow, At, LocationWord, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, At, LocationWord, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, ThereTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, ThereTemplateCoref],
    [Human, Place, RepeatCount, DescribingWord, InARow, HereTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, HereTemplateCoref],
    [Human, Place, RepeatCount, DescribingWord, InARow, YouTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Place N blocks in a row at every location Y ##
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build X out of Y with size ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],

    ## Build N X out of Y with size ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],

    ## Build n by m X with other size attributes like thickness etc ##
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],

    ## Build wall X blocks long and Y blocks high ##
    [Human, Build, Wall, NumBlocks, Squares, Long, And, NumBlocks, Squares, High],
    [Human, Build, Wall, NumBlocks, Squares, High, And, NumBlocks, Squares, Long],
    [Human, Build, Wall, NumBlocks, Squares, Long, NumBlocks, Squares, High],
    [Human, Build, Wall, NumBlocks, Squares, High, NumBlocks, Squares, Long],

    ## Build N walls X blocks long and Y blocks high ##
    [Human, Build, RepeatCount, Wall, NumBlocks, Squares, Long, And, NumBlocks, Squares, High],
    [Human, Build, RepeatCount, Wall, NumBlocks, Squares, High, And, NumBlocks, Squares, Long],
    [Human, Build, RepeatCount, Wall, NumBlocks, Squares, Long, NumBlocks, Squares, High],
    [Human, Build, RepeatCount, Wall, NumBlocks, Squares, High, NumBlocks, Squares, Long],

    ## Build X blocks long and Y blocks high wall ##
    [Human, Build, NumBlocks, Squares, Long, And, NumBlocks, Squares, High, Wall],
    [Human, Build, NumBlocks, Squares, High, And, NumBlocks, Squares, Long, Wall],
    [Human, Build, NumBlocks, Squares, Long, NumBlocks, Squares, High, Wall],
    [Human, Build, NumBlocks, Squares, High, NumBlocks, Squares, Long, Wall],

    ## Build n by m X using block type Y with size attributes ##
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],

    ## Build N n by m Xs using block type Y with size attributes ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],

    ## Build X of size Y using block type Z ##
    [Human, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## Build n by m of size Y using block type Z ##
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## out of X build Y ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicSize, DescribingWord],

    ## Build X out of block type Y ##
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType],

    ## Build n Xs with size Y out of block type Z ##
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## Out of X build n Ys ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicSize, DescribingWord],

    ## Build N Xs out of Y ##
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType],

    ## Build X with size Y ##
    [Human, Build, DescribingWord, WithAttributes],
    [Human, Build, SchematicColour, DescribingWord, WithAttributes],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Build, SchematicsDimensions, SchematicColour, DescribingWord, WithAttributes],

    ## Build X n times ###
    [Human, Build, DescribingWord, NTimes],
    [Human, Build, SchematicSize, DescribingWord, NTimes],
    [Human, Build, SchematicColour, DescribingWord, NTimes],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, NTimes],

    # Build size colour X ##
    [Human, Build, DescribingWord],
    [Human, Build, SchematicSize, DescribingWord],
    [Human, Build, SchematicColour, DescribingWord],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord],

    ## Build a block type X Y ##
    [Human, Build, UsingBlockType, DescribingWord],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord],

    ## Build N X of size Y ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Build, RepeatCount, SchematicsDimensions, SchematicColour, DescribingWord, WithAttributes],

    ## Build N X ##
    [Human, Build, RepeatCount, DescribingWord],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    ## Build N block type X Y ##
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    ## Out of X build Y with size Z at location A ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ThereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, HereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Out of X build Y with size Z at every location A ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Out of C build a by b X with size Y at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Out of C build a by b X with size Y at every location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Out of X build N Ys with size Z at location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ThereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, HereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Out of X build N Ys with size Z at every location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Out of X build N a by b Ys  with size Z at location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Out of X build N a by b Ys  with size Z at every location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Out of X build Y at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicSize, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build X of size Y at location Z ##
    [Human, Build, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicColour, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, SchematicColour, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build X at location Y ##
    [Human, Build, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicColour, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    ## Out of X build N Ys at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicSize, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ThereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ThereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, HereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, HereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Out of X build Y at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ThereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ThereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, HereTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, HereTemplateCoref],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Out of X build Y at every location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build N Ys with size X at location Z ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, SchematicColour, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, At, LocationWord, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build N Y with size X at every location Z ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build X with size Y at every location Z ##
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build X with size Y at location Z ##

    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, DescribingWord, ThereTemplate],
    [Human, Build, SchematicSize, DescribingWord, ThereTemplate],
    [Human, Build, SchematicColour, DescribingWord, ThereTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ThereTemplate],

    [Human, Build, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, DescribingWord, ThereTemplateCoref],
    [Human, Build, SchematicSize, DescribingWord, ThereTemplateCoref],
    [Human, Build, SchematicColour, DescribingWord, ThereTemplateCoref],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ThereTemplateCoref],

    [Human, Build, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, DescribingWord, HereTemplate],
    [Human, Build, SchematicSize, DescribingWord, HereTemplate],
    [Human, Build, SchematicColour, DescribingWord, HereTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, HereTemplate],

    [Human, Build, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, DescribingWord, HereTemplateCoref],
    [Human, Build, SchematicSize, DescribingWord, HereTemplateCoref],
    [Human, Build, SchematicColour, DescribingWord, HereTemplateCoref],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, HereTemplateCoref],

    [Human, Build, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, DescribingWord, YouTemplate],
    [Human, Build, SchematicSize, DescribingWord, YouTemplate],
    [Human, Build, SchematicColour, DescribingWord, YouTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, YouTemplate],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    ## Build N X with size Y at location Z ##

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, RepeatCount, DescribingWord, ThereTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ThereTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ThereTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ThereTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, RepeatCount, DescribingWord, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ThereTemplateCoref],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, RepeatCount, DescribingWord, HereTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, HereTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, HereTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, HereTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, RepeatCount, DescribingWord, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, HereTemplateCoref],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, RepeatCount, DescribingWord, YouTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, YouTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, YouTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, YouTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, YouTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    ## Build N X with size Y at every location Z ##

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## At location Y build X using block type A with size Z ##
    [Human, At, LocationWord, CoordinatesTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, At, LocationWord, CoordinatesTemplate, Build, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicSize, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicColour, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicSize, SchematicColour, DescribingWord],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, SchematicColour, DescribingWord],

    ## At location X build Y Z with size A ##
    [Human, At, LocationWord, CoordinatesTemplate, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, UsingBlockType, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicSize, UsingBlockType, DescribingWord],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, UsingBlockType, DescribingWord],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, UsingBlockType, DescribingWord],

    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicSize, UsingBlockType, DescribingWord],

    ## At location Y build N Xs using block type A with size Z ##
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    ## At every location Y build N Xs out of Z with size A  ##
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicColour, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicColour, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, SchematicColour, DescribingWord, RepeatAllLocation],

    ## At location X build Y out of Z with size A ##
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, SchematicColour, DescribingWord],

    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicSize, SchematicColour, DescribingWord],

    ## At location X build N Ys out of Z with size A ##
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    ## At every location X build N Ys out of Z with size A ##
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicColour, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RepeatAllLocation],

    ## At every location X build Y out of Z with size A ##
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicColour, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, SchematicColour, DescribingWord, RepeatAllLocation],

    ## Build X out of Y with size Z at location A ##
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplateCoref],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplateCoref],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplateCoref],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplateCoref],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, HereTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplateCoref],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplateCoref],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, HereTemplateCoref],

    ## Build n Xs out of Y with size Z at location A ##
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ThereTemplateCoref],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ThereTemplateCoref],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, HereTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, HereTemplateCoref],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, HereTemplateCoref],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, YouTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, YouTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, YouTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, YouTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build n Xs out of Y with size Z at every location A ##
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build X out of Y with size Z at every location A ##
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build X out Y with size Z at location A ##
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, YouTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, YouTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, YouTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, YouTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    ## build X Y with size Z at location A ##
    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, UsingBlockType, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, At, LocationWord, CoordinatesTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, UsingBlockType, DescribingWord, ThereTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ThereTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, UsingBlockType, DescribingWord, ThereTemplateCoref],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ThereTemplateCoref],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, UsingBlockType, DescribingWord, HereTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, HereTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, UsingBlockType, DescribingWord, HereTemplateCoref],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, HereTemplateCoref],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, UsingBlockType, DescribingWord, YouTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, YouTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    ## At location A build N X Y with size Z ##
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    ## At every location X build X Y with size Z  ##
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, UsingBlockType, DescribingWord, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, UsingBlockType, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, UsingBlockType, DescribingWord, RepeatAllLocation],

    ## At every location X build n X Ys with size Z  ##
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RepeatAllLocation],

    ## Build N X Y with size Z at location A ##
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, At, LocationWord, CoordinatesTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    ## Build X Y with size Z at every location A ##
    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    ## Build N X Y with size Z at location A ##

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ThereTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ThereTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ThereTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ThereTemplateCoref],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ThereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ThereTemplateCoref],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, HereTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, HereTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, HereTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, HereTemplateCoref],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, HereTemplateCoref],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, YouTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, YouTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, YouTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build N X Y with size Z at every location A ##

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## At location X Build X with size Y out of Z ##
    [Human, At, LocationWord, CoordinatesTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    [Human, At, LocationWord, CoordinatesTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## At location X Build n Xs with size Y out of Z ##
    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    [Human, At, LocationWord, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, CoordinatesTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## At every location X build n Xs with size Y out of Z ##
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],

    ## Build X with size Y out of Z at location A ##
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplateCoref],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, YouTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplateCoref],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, YouTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build n Xs with size Y out of Z at location A ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplateCoref],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, YouTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, At, LocationWord, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ThereTemplateCoref],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, HereTemplateCoref],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, YouTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, YouTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, CoordinatesTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate],

    ## Build n Xs with size Y out of Z at every location A ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Build X with size Y out of Z at every location A ##
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatAllLocation],

    ## Place N blocks at n locations X ##
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    ## Out of X build Y with size Z at M locations A (same templates as above with N locations) ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicColour, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicColour, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, SchematicColour, DescribingWord, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicColour, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicColour, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, SchematicColour, DescribingWord, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicSize, UsingBlockType, DescribingWord, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, UsingBlockType, DescribingWord, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicSize, UsingBlockType, DescribingWord, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],

    [Human, RelativeDirectionTemplate, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],
    [Human, RelativeDirectionTemplate, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationBlockObjectTemplate, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, LocationMobTemplate, RepeatCountLocation],

    # Build N X Around Y ##
    [Human, Build, RepeatCount, DescribingWord, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, Around, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Around, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, Around, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Around, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, Around, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Around, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, Around, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Around, BlockObjectThat],

    ## Build Xs around Y ##
    [Human, Build, RepeatAll, DescribingWord, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatAll, DescribingWord, WithAttributes, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatAll, DescribingWord, MadeOutOf, UsingBlockType, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatAll, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Around, LocationBlockObjectTemplate],
    [Human, Build, RepeatAll, DescribingWord, Around, LocationMobTemplate],
    [Human, Build, RepeatAll, DescribingWord, WithAttributes, Around, LocationMobTemplate],
    [Human, Build, RepeatAll, DescribingWord, MadeOutOf, UsingBlockType, Around, LocationMobTemplate],
    [Human, Build, RepeatAll, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Around, LocationMobTemplate],
    [Human, Build, RepeatAll, DescribingWord, Around, BlockObjectThat],
    [Human, Build, RepeatAll, DescribingWord, WithAttributes, Around, BlockObjectThat],
    [Human, Build, RepeatAll, DescribingWord, MadeOutOf, UsingBlockType, Around, BlockObjectThat],
    [Human, Build, RepeatAll, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Around, BlockObjectThat],

    ## Templates for rel_dir of BlockObjectThat ##
    [Human, BuildSingle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, BuildSingle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    # Build at n locations Y ##
    [Human, BuildSingle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Stack N blocks at every location Y ##
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Stack N blocks at M locations Y ##
    [Human, Stack, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Stack, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Place N blocks in a row at location Y ##
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Place N blocks at every location Y ##
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Place N blocks in a row at location Y ##
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Place N blocks in a row at every location Y ##
    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Out of X build Y with size Z at location A ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Out of X build Y with size Z at every location A ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Out of C build a by b X with size Y at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Out of C build a by b X with size Y at every location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Out of X build N Ys with size Z at location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Out of X build N Ys with size Z at every location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Out of X build N a by b Ys  with size Z at location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Out of X build N a by b Ys  with size Z at every location A ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Out of X build Y at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],

    ## Build X of size Y at location Z ##
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],

    ## Build X at location Y ##
    [Human, Build, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],

    ## Out of X build N Ys at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Out of X build Y at location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Out of X build Y at every location Z ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],

    ## Build N Y with size X at every location Z ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build X with size Y at every location Z ##
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build X with size Y at location Z ##

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Build N X with size Y at location Z ##

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Build N X with size Y at every location Z ##

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## At location Y build X using block type A with size Z ##

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicSize, SchematicColour, DescribingWord],

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicSize, UsingBlockType, DescribingWord],

    ## At location Y build N Xs using block type A with size Z ##

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    ## At every location Y build N Xs out of Z with size A  ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicSize, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicColour, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, MadeOutOf, UsingBlockType, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicSize, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicColour, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicSize, SchematicColour, DescribingWord, RepeatAllLocation],

    ## At location X build N Ys out of Z with size A ##

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Build n Xs out of Y with size Z at every location A ##
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build X out of Y with size Z at every location A ##
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build X out Y with size Z at location A ##
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## At location A build N X Y with size Z ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    ## At every location X build X Y with size Z  ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, UsingBlockType, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicSize, UsingBlockType, DescribingWord, RepeatAllLocation],

    ## At every location X build n X Ys with size Z  ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, UsingBlockType, DescribingWord, RepeatAllLocation],
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RepeatAllLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build X Y with size Z at every location A ##
    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build N X Y with size Z at location A ##

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## At location X Build X with size Y out of Z ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## At location X Build n Xs with size Y out of Z ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    ## At every location X build n Xs with size Y out of Z ##
    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],

    [Human, RelativeDirectionTemplate, BlockObjectThat, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RepeatAllLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],


    ## Build X with size Y out of Z at location A ##
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Build n Xs with size Y out of Z at location A ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat],

    ## Build n Xs with size Y out of Z at every location A ##
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Build X with size Y out of Z at every location A ##
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatAllLocation],

    ## Place N blocks at n locations X ##
    [Human, Place, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Place, RepeatCount, DescribingWord, InARow, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Place, RepeatCount, DescribingWord, InARow, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    ## Out of X build Y with size Z at M locations A (same templates as above with N locations) ##
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],


    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, StepsTemplate, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, ALittle, RelativeDirectionTemplate, BlockObjectThat, RepeatCountLocation],

    # Between templates
    [Human, BuildSingle, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Stack, RepeatCount, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, StepsTemplate, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, UsingBlockType, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicSize, UsingBlockType, DescribingWord],

    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicSize, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicColour, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicSize, SchematicColour, DescribingWord],

    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, UsingBlockType, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Between, LocationMobTemplate, And, LocationBlockObjectTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationMobTemplate, And, LocationBlockObjectTemplate],

    [Human, BuildSingle, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, BuildSingle, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Stack, RepeatCount, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Place, RepeatCount, DescribingWord, InARow, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, MadeOutOf, UsingBlockType, Build, RepeatCount, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, MadeOutOf, UsingBlockType, Build, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, DescribingWord, WithAttributes, StepsTemplate, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, StepsTemplate, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, DescribingWord, StepsTemplate, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, StepsTemplate, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicColour, DescribingWord, StepsTemplate, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, SchematicColour, DescribingWord, StepsTemplate, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, UsingBlockType, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicSize, UsingBlockType, DescribingWord],

    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicSize, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicColour, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicSize, SchematicColour, DescribingWord],

    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicSize, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicColour, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicSize, SchematicColour, DescribingWord],

    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, MadeOutOf, UsingBlockType, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, DescribingWord, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Build, UsingBlockType, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, UsingBlockType, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicSize, UsingBlockType, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, UsingBlockType, DescribingWord],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord],

    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, UsingBlockType, DescribingWord, WithAttributes, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, UsingBlockType, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicSize, UsingBlockType, DescribingWord, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],

    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],
    [Human, Between, LocationBlockObjectTemplate, And, LocationMobTemplate, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType],

    [Human, Build, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    [Human, Build, RepeatCount, SchematicsDimensions, DescribingWord, WithAttributes, MadeOutOf, UsingBlockType, Between, LocationBlockObjectTemplate, And, LocationMobTemplate],
    ]
