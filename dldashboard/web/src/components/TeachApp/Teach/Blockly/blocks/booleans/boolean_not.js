/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that inverts the value of a given boolean.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const notJSON = {
  type: "boolean_not",
  message0: "not %1",
  args0: [
    {
      type: "input_value",
      name: "A",
      check: [types.Boolean],
    },
  ],
  inputsInline: true,
  output: types.Boolean,
  style: "logic_blocks",
  helpUrl: "%{BKY_LOGIC_COMPARE_HELPURL}",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_not"] = {
  init: function () {
    this.jsonInit(notJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_not"] = function (block) {
  const a = Blockly.JavaScript.valueToCode(
    block,
    "A",
    Blockly.JavaScript.ORDER_NONE
  );
  const dict = `{ "condition": {
    "condition_type": "NOT",
    "not_condition": ${a}
  }}`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
