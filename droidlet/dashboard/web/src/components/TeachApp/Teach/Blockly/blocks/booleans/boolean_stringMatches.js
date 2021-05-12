/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that applies a mathematical binary comparator to two numbers.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const comparatorJSON = {
  type: "boolean_stringMatches",
  message0: "%1 text matches %2",
  args0: [
    {
      type: "input_value",
      name: "A",
      check: [types.String],
    },
    {
      type: "input_value",
      name: "B",
      check: [types.String],
    },
  ],
  inputsInline: true,
  output: types.Boolean,
  style: "logic_blocks",
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_stringMatches"] = {
  init: function () {
    this.jsonInit(comparatorJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_stringMatches"] = function (block) {
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
  const dict = `{
    "condition": {
      "condition_type": "COMPARATOR",
      "input_left": { "value_extractor": ${a} },
      "comparison_type": "EQUAL",
      "input_right": {
        "value_extractor": ${b}
      }
    }
  }`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
