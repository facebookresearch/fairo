"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# fmt: off
'''
Every template contains an ordered list of TemplateObjects.
TemplateObject is defined in template_objects.py

Destroy templates are written only for BlockObject and represent the intent
for the action: Destroy. This action destroys a physical block object.

Examples:
[Human, Destroy, The, Colour, AbstractDescription, BlockObjectLocation]
- destroy the red structure to the left of that grey shape.
- destroy the blue thing at location: 2 , 3, 4
- remove the blue shape there

[Human, Destroy, BlockObjectThat, Size, Colour, AbstractDescription]
- remove that huge red structure
- dig that tiny blue thing
'''


from base_agent.ttad.generation_dialogues.template_objects import *

DESTROY_WITH_CORRECTION = [
    ## Destroy this / that X ##
    [[Human, Destroy, BlockObjectThat],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectIt],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThat, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThat, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    ## Destroy the X ##
    [[Human, Destroy, The, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, The, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    ## Destroy this / that colour X ##
    [[Human, Destroy, BlockObjectThat, Colour, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis, Colour, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, The, Colour, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThat, Colour, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis, Colour, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, The, Colour, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],

    # ## Destroy this / that size X ##
    [[Human, Destroy, BlockObjectThat, Size, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis, Size, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, The, Size, AbstractDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThat, Size, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, BlockObjectThis, Size, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
    [[Human, Destroy, The, Size, ConcreteDescription],
     [HumanReplace, The, AbstractDescription, BlockObjectLocation]],
]

DESTROY_TEMPLATES = [
    ## Single word destroy commands ##
    [Human, DestroySingle],

    ## Destroy everything ##
    [Human, DestroySingle, RepeatAll],

    ## Destroy what you built ##
    # NOTE: this is only in destroy right now, but can be extended.
    [Human, Destroy, BlockObjectCoref],

    ## Destroy this / that X ##
    [Human, Destroy, BlockObjectThat],
    [Human, Destroy, BlockObjectThis],
    [Human, Destroy, BlockObjectIt],
    [Human, Destroy, BlockObjectThat, AbstractDescription],
    [Human, Destroy, BlockObjectThis, AbstractDescription],
    [Human, Destroy, BlockObjectThat, ConcreteDescription],
    [Human, Destroy, BlockObjectThis, ConcreteDescription],

    ## Destroy the X ##
    [Human, Destroy, The, AbstractDescription],
    [Human, Destroy, The, ConcreteDescription],
    [Human, Destroy, AbstractDescription],
    [Human, Destroy, ConcreteDescription],

    ## Destroy the X at location Y ##
    [Human, Destroy, The, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, The, ConcreteDescription, BlockObjectLocation],

    ## Destroy this / that colour X ##
    [Human, Destroy, BlockObjectThat, Colour, AbstractDescription],
    [Human, Destroy, BlockObjectThis, Colour, AbstractDescription],
    [Human, Destroy, The, Colour, AbstractDescription],
    [Human, Destroy, BlockObjectThat, Colour, ConcreteDescription],
    [Human, Destroy, BlockObjectThis, Colour, ConcreteDescription],
    [Human, Destroy, The, Colour, ConcreteDescription],

    ## Destroy this / that colour X ##
    [Human, Destroy, BlockObjectThat, Size, AbstractDescription],
    [Human, Destroy, BlockObjectThis, Size, AbstractDescription],
    [Human, Destroy, The, Size, AbstractDescription],
    [Human, Destroy, BlockObjectThat, Size, ConcreteDescription],
    [Human, Destroy, BlockObjectThis, Size, ConcreteDescription],
    [Human, Destroy, The, Size, ConcreteDescription],

    ## Destroy the size X at location Y ##
    [Human, Destroy, The, Size, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, The, Size, ConcreteDescription, BlockObjectLocation],

    ## Destroy the colour X at location Y ##
    [Human, Destroy, The, Colour, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, The, Colour, ConcreteDescription, BlockObjectLocation],

    ## Destroy the size colour X ##
    [Human, Destroy, BlockObjectThat, Size, Colour, AbstractDescription],
    [Human, Destroy, BlockObjectThis, Size, Colour, AbstractDescription],
    [Human, Destroy, The, Size, Colour, AbstractDescription],
    [Human, Destroy, BlockObjectThat, Size, Colour, ConcreteDescription],
    [Human, Destroy, BlockObjectThis, Size, Colour, ConcreteDescription],
    [Human, Destroy, The, Size, Colour, ConcreteDescription],

    ## Destroy the size colour X at location Y ##
    [Human, Destroy, The, Size, Colour, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, The, Size, Colour, ConcreteDescription, BlockObjectLocation],

    ## Destroy num X ##
    [Human, Destroy, RepeatCount, AbstractDescription],
    [Human, Destroy, RepeatCount, ConcreteDescription],

    ## Destroy num colour X ##
    [Human, Destroy, RepeatCount, Colour, AbstractDescription],
    [Human, Destroy, RepeatCount, Colour, ConcreteDescription],

    ## Destroy num size X ##
    [Human, Destroy, RepeatCount, Size, AbstractDescription],
    [Human, Destroy, RepeatCount, Size, ConcreteDescription],

    ## Destroy num size colour X ##
    [Human, Destroy, RepeatCount, Size, Colour, AbstractDescription],
    [Human, Destroy, RepeatCount, Size, Colour, ConcreteDescription],

    ## Destroy num size X at location Y ##
    [Human, Destroy, RepeatCount, Size, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, RepeatCount, Size, ConcreteDescription, BlockObjectLocation],

    ## Destroy num colour X at location Y ##
    [Human, Destroy, RepeatCount, Colour, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation],

    ## Destroy num size colour X at location Y ##
    [Human, Destroy, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation],
    [Human, Destroy, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation],

    ### Destroy X at locations Y ###
    [Human, Destroy, The, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, The, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    [Human, Destroy, The, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, The, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    [Human, Destroy, The, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, The, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    [Human, Destroy, The, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, The, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    ### Destroy num X at locations Y ###
    [Human, Destroy, RepeatCount, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, RepeatCount, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    [Human, Destroy, RepeatCount, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    [Human, Destroy, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    [Human, Destroy, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation],
    [Human, Destroy, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation],

    ### Destroy all X at location Y ###
    [Human, Destroy, All, AbstractDescription, RepeatAll],
    [Human, Destroy, Every, AbstractDescription, RepeatAll],
    [Human, Destroy, All, ConcreteDescription, RepeatAll],
    [Human, Destroy, Every, ConcreteDescription, RepeatAll],

    [Human, Destroy, All, Colour, ConcreteDescription, RepeatAll],
    [Human, Destroy, Every, Colour, ConcreteDescription, RepeatAll],
    [Human, Destroy, All, Colour, AbstractDescription, RepeatAll],
    [Human, Destroy, Every, Colour, AbstractDescription, RepeatAll],

    [Human, Destroy, All, Size, AbstractDescription, RepeatAll],
    [Human, Destroy, Every, Size, AbstractDescription, RepeatAll],
    [Human, Destroy, All, Size, ConcreteDescription, RepeatAll],
    [Human, Destroy, Every, Size, ConcreteDescription, RepeatAll],

    [Human, Destroy, All, Size, AbstractDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, All, Size, ConcreteDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, Every, Size, AbstractDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, Every, Size, ConcreteDescription, BlockObjectLocation, RepeatAll],

    [Human, Destroy, All, Colour, AbstractDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, All, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, Every, Colour, AbstractDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, Every, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll],

    [Human, Destroy, All, Size, Colour, AbstractDescription, RepeatAll],
    [Human, Destroy, Every, Size, Colour, AbstractDescription, RepeatAll],
    [Human, Destroy, All, Size, Colour, ConcreteDescription, RepeatAll],
    [Human, Destroy, Every, Size, Colour, ConcreteDescription, RepeatAll],

    [Human, Destroy, All, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, All, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, Every, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAll],
    [Human, Destroy, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAll],

    ## Destroy X at every location Y ##
    [Human, Destroy, The, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, The, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    [Human, Destroy, The, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, The, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    [Human, Destroy, The, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, The, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    [Human, Destroy, The, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, The, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    ### Destroy all X at every location Y ###
    [Human, Destroy, All, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, All, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],

    [Human, Destroy, All, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, All, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],

    [Human, Destroy, All, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, All, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],

    [Human, Destroy, All, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, All, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],
    [Human, Destroy, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation, RepeatAll],

    ### Destroy all X at locations Ys ###
    [Human, Destroy, All, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, All, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],

    [Human, Destroy, All, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, All, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, Size, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, Size, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],

    [Human, Destroy, All, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, All, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],

    [Human, Destroy, All, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, All, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],
    [Human, Destroy, Every, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatCountLocation, RepeatAll],

    ### Destroy n X at every location Y ###
    [Human, Destroy, RepeatCount, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, RepeatCount, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    [Human, Destroy, RepeatCount, Size, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, RepeatCount, Size, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    [Human, Destroy, RepeatCount, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, RepeatCount, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],

    [Human, Destroy, RepeatCount, Size, Colour, AbstractDescription, BlockObjectLocation, RepeatAllLocation],
    [Human, Destroy, RepeatCount, Size, Colour, ConcreteDescription, BlockObjectLocation, RepeatAllLocation],
    ]
