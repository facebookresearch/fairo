"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
"""
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

GetMemory templates are written for filters and have an answer_type
They represent the action of fetching from the memory using the filters.

Examples:

[Human, QueryBotCurrentAction],
- human: what are you doing
- human: what are you up to

[Human, QueryBot, MoveTarget],
- human: where you going
- human: where are you heading
"""

ANSWER_WITH_CORRECTION = [
    ## what is this + the thing at location ##
    [[Human, What, Is, BlockObjectThis],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, What, Is, BlockObjectThis, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, What, Is, BlockObjectThat],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, What, Is, BlockObjectThat, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

     ## what size is X + the thing at location ##
     [[Human, AskSize, BlockObjectThis],
      [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
     [[Human, AskSize, BlockObjectThis, AbstractDescription],
      [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskSize, BlockObjectThis, ConcreteDescription],
      [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
     [[Human, AskSize, BlockObjectThat],
      [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
     [[Human, AskSize, BlockObjectThat, AbstractDescription],
      [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
     [[Human, AskSize, BlockObjectThat, ConcreteDescription],
      [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    ## what color is X + the thing at location ##
    [[Human, AskColour, BlockObjectThis],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskColour, BlockObjectThis, AbstractDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskColour, BlockObjectThis, ConcreteDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskColour, BlockObjectThat],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskColour, BlockObjectThat, AbstractDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskColour, BlockObjectThat, ConcreteDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    # Is X Y ##
    [[Human, AskIs, BlockObjectThis, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThis, AbstractDescription, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThis, ConcreteDescription, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, AbstractDescription, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, ConcreteDescription, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, The, AbstractDescription, BlockObjectLocation, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, The, ConcreteDescription, BlockObjectLocation, Size],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    [[Human, AskIs, BlockObjectThis, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThis, AbstractDescription, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThis, ConcreteDescription, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, AbstractDescription, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, ConcreteDescription, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, The, AbstractDescription, BlockObjectLocation, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, The, ConcreteDescription, BlockObjectLocation, Colour],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    ## Is X a Y ##
    [[Human, AskIs, BlockObjectThis, ConcreteDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThis, AbstractDescription, ConcreteDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, ConcreteDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, AskIs, BlockObjectThat, AbstractDescription, ConcreteDescription],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

]

ANSWER_TEMPLATES = [
    # 1
    ## What is X ##
    [Human, What, Is, BlockObjectThis],
    [Human, What, Is, BlockObjectThis, AbstractDescription],
    [Human, What, Is, BlockObjectThat],
    [Human, What, Is, BlockObjectThat, AbstractDescription],

    # 2
    ## What is at X ##
    [Human, What, Is, BlockObjectLocation],
    [Human, What, Is, The, AbstractDescription, BlockObjectLocation],

    ## What do you see at X ##
    [Human, WhatSee, BlockObjectLocation],

    # 3
    ## What size is X ##
    [Human, AskSize, BlockObjectThis],
    [Human, AskSize, BlockObjectThis, AbstractDescription],
    [Human, AskSize, BlockObjectThis, ConcreteDescription],
    [Human, AskSize, BlockObjectThat],
    [Human, AskSize, BlockObjectThat, AbstractDescription],
    [Human, AskSize, BlockObjectThat, ConcreteDescription],

    # 4
    ## what size is X at Y ##
    [Human, AskSize, The, AbstractDescription, BlockObjectLocation],
    [Human, AskSize, The, ConcreteDescription, BlockObjectLocation],

    # 5
    # What colour is X ##
    [Human, AskColour, BlockObjectThis],
    [Human, AskColour, BlockObjectThis, AbstractDescription],
    [Human, AskColour, BlockObjectThis, ConcreteDescription],
    [Human, AskColour, BlockObjectThat],
    [Human, AskColour, BlockObjectThat, AbstractDescription],
    [Human, AskColour, BlockObjectThat, ConcreteDescription],

    # 6
    ## what colour is X at Y ##
    [Human, AskColour, The, AbstractDescription, BlockObjectLocation],
    [Human, AskColour, The, ConcreteDescription, BlockObjectLocation],

    # 7
    ## Is X Y ##
    [Human, AskIs, BlockObjectThis, Size],
    [Human, AskIs, BlockObjectThis, AbstractDescription, Size],
    [Human, AskIs, BlockObjectThis, ConcreteDescription, Size],
    [Human, AskIs, BlockObjectThat, Size],
    [Human, AskIs, BlockObjectThat, AbstractDescription, Size],
    [Human, AskIs, BlockObjectThat, ConcreteDescription, Size],

    [Human, AskIs, The, AbstractDescription, BlockObjectLocation, Size],
    [Human, AskIs, The, ConcreteDescription, BlockObjectLocation, Size],

    [Human, AskIs, BlockObjectThis, Colour],
    [Human, AskIs, BlockObjectThis, AbstractDescription, Colour],
    [Human, AskIs, BlockObjectThis, ConcreteDescription, Colour],
    [Human, AskIs, BlockObjectThat, Colour],
    [Human, AskIs, BlockObjectThat, AbstractDescription, Colour],
    [Human, AskIs, BlockObjectThat, ConcreteDescription, Colour],

    [Human, AskIs, The, AbstractDescription, BlockObjectLocation, Colour],
    [Human, AskIs, The, ConcreteDescription, BlockObjectLocation, Colour],

    # 8
    ## Is X a Y ##
    [Human, AskIs, BlockObjectThis, ConcreteDescription],
    [Human, AskIs, BlockObjectThis, AbstractDescription, ConcreteDescription],
    [Human, AskIs, BlockObjectThat, ConcreteDescription],
    [Human, AskIs, BlockObjectThat, AbstractDescription, ConcreteDescription],

    # 9
    ## IS X at Y Z ##
    [Human, AskIs, The, AbstractDescription, BlockObjectLocation, ConcreteDescription],

] 

GET_MEMORY_TEMPLATES = [
    ## What are you Doing (Action name) ##
    [Human, QueryBotCurrentAction],

    ## What are you Building (Action reference object name) ##
    [Human, QueryBot, ActionReferenceObjectName],

    ## Where are you heading (Move target) ##
    [Human, QueryBot, MoveTarget],

    ## Where are you (Bot location) ##
    [Human, QueryBot, CurrentLocation],
] + ANSWER_TEMPLATES
