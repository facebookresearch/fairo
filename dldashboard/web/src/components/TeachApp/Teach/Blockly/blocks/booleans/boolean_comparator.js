/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that applies a mathematical binary comparator to two numbers.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import { generate_dropdown_comparator } from "../fields/dropdowns";
import customInit from "../utils/customInit";

const comparatorJSON = {
  type: "boolean_comparator",
  message0: "%1 %2 %3",
  args0: [
    {
      type: "input_value",
      name: "A",
      check: [types.Number],
    },
    generate_dropdown_comparator("COMP"),
    {
      type: "input_value",
      name: "B",
      check: [types.Number],
    },
  ],
  inputsInline: true,
  output: types.Boolean,
  style: "logic_blocks",
  helpUrl: "%{BKY_LOGIC_COMPARE_HELPURL}",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_comparator"] = {
  init: function () {
    this.jsonInit(comparatorJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_comparator"] = function (block) {
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
  const comparator = block.getFieldValue("COMP");
  const dict = `{
    "condition": {
      "condition_type": "COMPARATOR",
      "input_left": {
        "value_extractor": ${a}
      },
      "input_right": {
        "value_extractor": ${b}
      },
      "comparison_type": "${comparator}"
    }
  }`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
