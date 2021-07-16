/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block representing an absolute location in the agent's frame of reference.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const minecraftLocationJSON = {
  type: "location",
  message0: "x %1 %2 y %3 %4 z %5",
  args0: [
    {
      type: "field_number",
      name: "x",
      value: 0,
    },
    {
      type: "input_dummy",
    },
    {
      type: "field_number",
      name: "y",
      value: 0,
    },
    {
      type: "input_dummy",
    },
    {
      type: "field_number",
      name: "z",
      value: 0,
    },
  ],
  inputsInline: true,
  output: types.Location,
  colour: 230,
  tooltip:
    "A location in the current bot's coordinate system, given in x, y, and z coordinates.",
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["location"] = {
  init: function () {
    this.jsonInit(minecraftLocationJSON);
    customInit(this);
  },
};

Blockly.JavaScript["location"] = function (block) {
  var x = block.getFieldValue("x");
  var y = block.getFieldValue("y");
  var z = block.getFieldValue("z");
  const logicalForm = `"location": { "reference_object": { "special_reference": { "coordinate_span": "${x}, ${y}, ${z}"} } }`;
  return [logicalForm, Blockly.JavaScript.ORDER_NONE];
};
