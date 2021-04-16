/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that applies a logical or to two booleans.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const orJSON = {
  type: "boolean_or",
  message0: "%1 or %2",
  args0: [
    {
      type: "input_value",
      name: "A",
      check: [types.Boolean],
    },
    {
      type: "input_value",
      name: "B",
      check: [types.Boolean],
    },
  ],
  inputsInline: true,
  output: types.Boolean,
  style: "logic_blocks",
  helpUrl: "%{BKY_LOGIC_COMPARE_HELPURL}",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_or"] = {
  init: function () {
    this.jsonInit(orJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_or"] = function (block) {
  const a = Blockly.JavaScript.valueToCode(
    block,
    "A",
    Blockly.JavaScript.ORDER_NONE
  );
  const b = Blockly.JavaScript.valueToCode(
    block,
    "B",
    Blockly.JavaScript.ORDER_NONE
  );
  const dict = `{ "condition": {
    "condition_type": "OR",
    "or_condition": [
      ${a},
      ${b}
    ]
  }}`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
