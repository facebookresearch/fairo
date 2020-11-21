/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that represents time passing in the game as a boolean.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import { generate_dropdown_timeUnits } from "../fields/dropdowns";
import customInit from "../utils/customInit";

const timeEventJSON = {
  type: "boolean_timePassed",
  message0: "%1 %2 have passed",
  args0: [
    {
      type: "field_number",
      name: "COUNT",
      value: 1,
    },
    generate_dropdown_timeUnits("UNITS"),
  ],
  inputsInline: true,
  output: types.Boolean,
  style: "logic_blocks",
  tooltip:
    "Represents time passing. Returns true after the amount of time specified has passed.",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_timePassed"] = {
  init: function () {
    this.jsonInit(timeEventJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_timePassed"] = function (block) {
  const count = block.getFieldValue("COUNT");
  const units = block.getFieldValue("UNITS");
  const dict = `{
    "condition": {
      "condition_type": "TIME",
      "comparison_type": "GREATER_THAN",
      "input_right": {
        "value_extractor": "${count}"
      },
      "comparison_measure": "${units}"
    }
  }`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
