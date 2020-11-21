/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * A basic block representing a single number.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const numberJSON = {
  type: "value_number",
  message0: "%1",
  args0: [
    {
      type: "field_number",
      name: "NUM",
      value: 0,
    },
  ],
  output: types.Number,
  helpUrl: "%{BKY_MATH_NUMBER_HELPURL}",
  style: "math_blocks",
  tooltip: "%{BKY_MATH_NUMBER_TOOLTIP}",
  extensions: ["parent_tooltip_when_inline"],
  mutator: "labelMutator",
};

Blockly.Blocks["value_number"] = {
  init: function () {
    this.jsonInit(numberJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_number"] = function (block) {
  const num = block.getFieldValue("NUM");
  return [num.toString(), Blockly.JavaScript.ORDER_NONE];
};
