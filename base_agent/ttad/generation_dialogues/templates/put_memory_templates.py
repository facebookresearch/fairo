"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
"""
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

PutMemory templates are written for filters and have an answer_type
They represent the action of writing to the memory using the filters.

Examples:

[[HumanMemory, HumanPosReward],
[Bot, BotThank]],
- human: good job
  bot: thanks for letting me know
"""
from template_objects import *

TAG_WITH_CORRECTION = [
    ## Add location ##
    ## X is adjective ##
    [[Human, BlockObjectThat, Is, TagDesc],
     [HumanReplace, The, Size, Thing]],
    [[Human, BlockObjectThat, Is, TagDesc],
     [HumanReplace, The, Colour, Thing]],
    [[Human, BlockObjectThat, Is, TagDesc],
     [HumanReplace, The, Size, Colour, Thing]],

    [[Human, BlockObjectThis, Is, TagDesc],
     [HumanReplace, The, Size, Thing]],
    [[Human, BlockObjectThis, Is, TagDesc],
     [HumanReplace, The, Colour, Thing]],
    [[Human, BlockObjectThis, Is, TagDesc],
     [HumanReplace, The, Size, Colour, Thing]],

    [[Human, BlockObjectIt, Is, TagDesc],
     [HumanReplace, The, Size, Thing]],
    [[Human, BlockObjectIt, Is, TagDesc],
     [HumanReplace, The, Colour, Thing]],
    [[Human, BlockObjectIt, Is, TagDesc],
     [HumanReplace, The, Size, Colour, Thing]],

    [[Human, Tag, BlockObjectThat, With, TagDesc],
      [HumanReplace, The, Size, Thing]],
    [[Human, Tag, BlockObjectThat, With, TagDesc],
      [HumanReplace, The, Colour, Thing]],
    [[Human, Tag, BlockObjectThat, With, TagDesc],
       [HumanReplace, The, Size, Colour, Thing]],

    [[Human, Tag, BlockObjectThis, With, TagDesc],
      [HumanReplace, The, Size, Thing]],
    [[Human, Tag, BlockObjectThis, With, TagDesc],
      [HumanReplace, The, Colour, Thing]],
    [[Human, Tag, BlockObjectThis, With, TagDesc],
       [HumanReplace, The, Size, Colour, Thing]],

    [[Human, Tag, BlockObjectIt, With, TagDesc],
      [HumanReplace, The, Size, Thing]],
    [[Human, Tag, BlockObjectIt, With, TagDesc],
      [HumanReplace, The, Colour, Thing]],
    [[Human, Tag, BlockObjectIt, With, TagDesc],
       [HumanReplace, The, Size, Colour, Thing]],

    # ## X is name ##
    [[Human, BlockObjectThat, Is, TagName],
     [HumanReplace, The, Size, Thing]],
    [[Human, BlockObjectThat, Is, TagName],
     [HumanReplace, The, Colour, Thing]],
    [[Human, BlockObjectThat, Is, TagName],
     [HumanReplace, The, Size, Colour, Thing]],

    [[Human, BlockObjectThis, Is, TagName],
     [HumanReplace, The, Size, Thing]],
    [[Human, BlockObjectThis, Is, TagName],
     [HumanReplace, The, Colour, Thing]],
    [[Human, BlockObjectThis, Is, TagName],
     [HumanReplace, The, Size, Colour, Thing]],

    [[Human, BlockObjectIt, Is, TagName],
     [HumanReplace, The, Size, Thing]],
    [[Human, BlockObjectIt, Is, TagName],
     [HumanReplace, The, Colour, Thing]],
    [[Human, BlockObjectIt, Is, TagName],
     [HumanReplace, The, Size, Colour, Thing]],

    [[Human, BlockObjectThat, Is, TagDesc],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, BlockObjectThis, Is, TagDesc],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, BlockObjectIt, Is, TagDesc],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    [[Human, Tag, BlockObjectThat, With, TagDesc],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Tag, BlockObjectThis, With, TagDesc],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Tag, BlockObjectIt, With, TagDesc],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    ## X is name ##
    [[Human, BlockObjectThat, Is, TagName],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, BlockObjectThis, Is, TagName],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, BlockObjectIt, Is, TagName],
    [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
]


TAG_TEMPLATES = [
    # X is adjective ##
    [Human, BlockObjectThat, Is, TagDesc],
    [Human, BlockObjectThis, Is, TagDesc],
    [Human, BlockObjectIt, Is, TagDesc],
    [Human, Tag, BlockObjectThat, With, TagDesc],
    [Human, Tag, BlockObjectThis, With, TagDesc],
    [Human, Tag, BlockObjectIt, With, TagDesc],

    ## X is name ##
    [Human, BlockObjectThat, Is, TagName],
    [Human, BlockObjectThis, Is, TagName],
    [Human, BlockObjectIt, Is, TagName],

    ## The X at Y is adjective ##
    [Human, The, AbstractDescription, BlockObjectLocation, Is, TagDesc],
    [Human, BlockObjectThat, AbstractDescription, Is, TagDesc],
    [Human, BlockObjectThis, AbstractDescription, Is, TagDesc],
    [Human, The, ConcreteDescription, BlockObjectLocation, Is, TagDesc],
    [Human, BlockObjectThat, ConcreteDescription, Is, TagDesc],
    [Human, BlockObjectThis, ConcreteDescription, Is, TagDesc],

    ## The size X is adjective ##
    [Human, BlockObjectThat, Size, AbstractDescription, Is, TagDesc],
    [Human, BlockObjectThis, Size, AbstractDescription, Is, TagDesc],
    [Human, The, Size, AbstractDescription, BlockObjectLocation, Is, TagDesc],
    [Human, BlockObjectThat, Size, ConcreteDescription, Is, TagDesc],
    [Human, BlockObjectThis, Size, ConcreteDescription, Is, TagDesc],
    [Human, The, Size, ConcreteDescription, BlockObjectLocation, Is, TagDesc],

    ## The colour X is adjective ##
    [Human, BlockObjectThat, Colour, AbstractDescription, Is, TagDesc],
    [Human, BlockObjectThis, Colour, AbstractDescription, Is, TagDesc],
    [Human, The, Colour, AbstractDescription, BlockObjectLocation, Is, TagDesc],
    [Human, BlockObjectThat, Colour, ConcreteDescription, Is, TagDesc],
    [Human, BlockObjectThis, Colour, ConcreteDescription, Is, TagDesc],
    [Human, The, Colour, ConcreteDescription, BlockObjectLocation, Is, TagDesc],

    ## The size colour X is adjective ##
    [Human, BlockObjectThat, Size, Colour, AbstractDescription, Is, TagDesc],
    [Human, BlockObjectThis, Size, Colour, AbstractDescription, Is, TagDesc],
    [Human, The, Size, Colour, AbstractDescription, BlockObjectLocation, Is, TagDesc],
    [Human, BlockObjectThat, Size, Colour, ConcreteDescription, Is, TagDesc],
    [Human, BlockObjectThis, Size, Colour, ConcreteDescription, Is, TagDesc],
    [Human, The, Size, Colour, ConcreteDescription, BlockObjectLocation, Is, TagDesc],

    # Tag X with adjective ##
    [Human, Tag, BlockObjectThat, AbstractDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, AbstractDescription, With, TagDesc],
    [Human, Tag,The, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, BlockObjectThat, ConcreteDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, ConcreteDescription, With, TagDesc],
    [Human, Tag,The, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    ## Tag size X with adjective ##
    [Human, Tag, BlockObjectThat, Size, AbstractDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, Size, AbstractDescription, With, TagDesc],
    [Human, Tag, The, Size, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, BlockObjectThat, Size, ConcreteDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, Size, ConcreteDescription, With, TagDesc],
    [Human, Tag, The, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    ## Tag colour X with adjective ##
    [Human, Tag, BlockObjectThat, Colour, AbstractDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, Colour, AbstractDescription, With, TagDesc],
    [Human, Tag, The, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, BlockObjectThat, Colour, ConcreteDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, Colour, ConcreteDescription, With, TagDesc],
    [Human, Tag, The, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    ## Tag size colour X with adjective ##
    [Human, Tag, BlockObjectThat, Size, Colour, AbstractDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, Size, Colour, AbstractDescription, With, TagDesc],
    [Human, Tag, The, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, BlockObjectThat, Size, Colour, ConcreteDescription, With, TagDesc],
    [Human, Tag, BlockObjectThis, Size, Colour, ConcreteDescription, With, TagDesc],
    [Human, Tag, The, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    ## The mob is/ looks adjective ##
    [Human, The, MobName, Is, TagDesc],
    [Human, MobThis, MobName, Is, TagDesc],
    [Human, MobThat, MobName, Is, TagDesc],
    [Human, The, MobName, MobLocation, Is, TagDesc],

    ## Tag mob with adjective ##
    [Human, Tag, The, MobName, With, TagDesc],
    [Human, Tag, The, MobName, MobLocation, With, TagDesc],
    [Human, Tag, MobThis, MobName, With, TagDesc],
    [Human, Tag, MobThat, MobName, With, TagDesc],

    ## The X is/ looks like a name ##
    [Human, The, AbstractDescription, BlockObjectLocation, Is, TagName],
    [Human, BlockObjectThat, AbstractDescription, Is, TagName],
    [Human, BlockObjectThis, AbstractDescription, Is, TagName],
    [Human, The, ConcreteDescription, BlockObjectLocation, Is, TagName],
    [Human, BlockObjectThat, ConcreteDescription, Is, TagName],
    [Human, BlockObjectThis, ConcreteDescription, Is, TagName],

    ## The size X is/ looks like a name ##
    [Human, BlockObjectThat, Size, AbstractDescription, Is, TagName],
    [Human, BlockObjectThis, Size, AbstractDescription, Is, TagName],
    [Human, The, Size, AbstractDescription, BlockObjectLocation, Is, TagName],
    [Human, BlockObjectThat, Size, ConcreteDescription, Is, TagName],
    [Human, BlockObjectThis, Size, ConcreteDescription, Is, TagName],
    [Human, The, Size, ConcreteDescription, BlockObjectLocation, Is, TagName],

    ## The colour X is/ looks like a name ##
    [Human, BlockObjectThat, Colour, AbstractDescription, Is, TagName],
    [Human, BlockObjectThis, Colour, AbstractDescription, Is, TagName],
    [Human, The, Colour, AbstractDescription, BlockObjectLocation, Is, TagName],
    [Human, BlockObjectThat, Colour, ConcreteDescription, Is, TagName],
    [Human, BlockObjectThis, Colour, ConcreteDescription, Is, TagName],
    [Human, The, Colour, ConcreteDescription, BlockObjectLocation, Is, TagName],

    ## The size colour X is/ looks like a name ##
    [Human, BlockObjectThat, Size, Colour, AbstractDescription, Is, TagName],
    [Human, BlockObjectThis, Size, Colour, AbstractDescription, Is, TagName],
    [Human, The, Size, Colour, AbstractDescription, BlockObjectLocation, Is, TagName],
    [Human, BlockObjectThat, Size, Colour, ConcreteDescription, Is, TagName],
    [Human, BlockObjectThis, Size, Colour, ConcreteDescription, Is, TagName],
    [Human, The, Size, Colour, ConcreteDescription, BlockObjectLocation, Is, TagName],

    ## The mob is / looks like a name ##
    [Human, The, MobName, Is, TagName],
    [Human, MobThis, MobName, Is, TagName],
    [Human, MobThat, MobName, Is, TagName],
    [Human, The, MobName, MobLocation, Is, TagName],

    ### Tag all X as Y ###
    [Human, Tag, Everything, With, TagDesc, RepeatAll],

    [Human, Tag, All, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, All, ConcreteDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, ConcreteDescription, With, TagDesc, RepeatAll],

    [Human, Tag, All, Size, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, All, Size, ConcreteDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, ConcreteDescription, With, TagDesc, RepeatAll],

    [Human, Tag, All, Colour, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, All, Colour, ConcreteDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Colour, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Colour, ConcreteDescription, With, TagDesc, RepeatAll],

    [Human, Tag, All, Size, Colour, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, All, Size, Colour, ConcreteDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, Colour, AbstractDescription, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, Colour, ConcreteDescription, With, TagDesc, RepeatAll],

    [Human, Tag, MobName, With, TagDesc, RepeatAll],

    ### Tag all X at location Y as Z ###
    [Human, Tag, Everything, BlockObjectLocation, With, TagDesc, RepeatAll],

    [Human, Tag, All, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, All, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],

    [Human, Tag, All, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, All, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],

    [Human, Tag, All, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, All, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],

    [Human, Tag, All, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, All, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAll],
    [Human, Tag, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAll],

    [Human, Tag, MobName, MobLocation, With, TagDesc, RepeatAll],

    ## Tag X at all Y as Z ##
    [Human, Tag, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, MobName, MobLocation, With, TagDesc, RepeatAllLocation],

    ## Tag all X at all Y as Z ##
    [Human, Tag, All, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, All, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, All, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, All, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, All, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, All, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, All, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, All, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, Every, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, Every, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, Every, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, Every, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, Every, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, Every, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, Every, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],
    [Human, Tag, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    [Human, Tag, MobName, MobLocation, With, TagDesc, RepeatAllLocation, RepeatAll],

    ### Tag num X as Y ###
    [Human, Tag, RepeatCount, AbstractDescription, With, TagDesc],
    [Human, Tag, RepeatCount, ConcreteDescription, With, TagDesc],

    [Human, Tag, RepeatCount, Size, AbstractDescription, With, TagDesc],
    [Human, Tag, RepeatCount, Size, ConcreteDescription, With, TagDesc],

    [Human, Tag, RepeatCount, Colour, AbstractDescription, With, TagDesc],
    [Human, Tag, RepeatCount, Colour, ConcreteDescription, With, TagDesc],

    [Human, Tag, RepeatCount, Size, Colour, AbstractDescription, With, TagDesc],
    [Human, Tag, RepeatCount, Size, Colour, ConcreteDescription, With, TagDesc],

    [Human, Tag, RepeatCount, MobName, With, TagDesc],

    ## Tag num X at location Y as Z ##
    [Human, Tag, RepeatCount, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, RepeatCount, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    [Human, Tag, RepeatCount, Size, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    [Human, Tag, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    [Human, Tag, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc],
    [Human, Tag, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc],

    [Human, Tag, RepeatCount, MobName, MobLocation, With, TagDesc],

    ## Tag X at num Y as Z ##
    [Human, Tag, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, MobName, MobLocation, With, TagDesc, RepeatCountLocation],

    ## Tag num X at num Y as Z ##
    [Human, Tag, RepeatCount, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, RepeatCount, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, RepeatCount, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],
    [Human, Tag, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation],

    [Human, Tag, RepeatCount, MobName, MobLocation, With, TagDesc, RepeatCountLocation],

    ## Tag all X at num Y as Z ##
    [Human, Tag, All, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, All, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, All, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, All, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, All, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, All, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, All, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, All, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, Every, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, Every, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, Every, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, Every, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, Every, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, Every, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, Every, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],
    [Human, Tag, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    [Human, Tag, MobName, MobLocation, With, TagDesc, RepeatCountLocation, RepeatAll],

    ## Tag num X at all Y as Z ##
    [Human, Tag, RepeatCount, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, RepeatCount, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, RepeatCount, Size, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],
    [Human, Tag, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, With, TagDesc, RepeatAllLocation],

    [Human, Tag, RepeatCount, MobName, MobLocation, With, TagDesc, RepeatAllLocation]
]


PUT_MEMORY_TEMPLATES = [
    ## Give positive reward ##
    [Human, HumanReward, PosReward],

    ## Give negative reward ##
    [Human, HumanReward, NegReward],

] + TAG_TEMPLATES
