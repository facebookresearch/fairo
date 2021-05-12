/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that applies a logical and to two booleans.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const andJSON = {
  type: "boolean_and",
  message0: "%1 and %2",
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

Blockly.Blocks["boolean_and"] = {
  init: function () {
    this.jsonInit(andJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_and"] = function (block) {
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
    "condition_type": "AND",
    "and_condition": [
      ${a},
      ${b}
    ]
  }}`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
