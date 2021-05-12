/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Filter option for filtering by reference object color.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../../utils/types";
import customInit from "../../utils/customInit";

const filterColorJSON = {
  type: "value_filterColor",
  message0: "has color %1",
  args0: [
    {
      type: "field_input",
      name: "COLOR",
      check: [types.String],
    },
  ],
  inputsInline: true,
  previousStatement: types.Filter,
  nextStatement: types.Filter,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Filter option for filtering by reference object color. Use with block object block.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_filterColor"] = {
  init: function () {
    this.jsonInit(filterColorJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_filterColor"] = function (block) {
  const color = block.getFieldValue("COLOR");
  const dict = `"has_color": "${color}",\n`;
  return dict;
};
