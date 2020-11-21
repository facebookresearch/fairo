/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that tells if two locations are related in a specified way.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import {
  generate_dropdown_comparator,
  generate_dropdown_distanceUnits,
} from "../fields/dropdowns";
import customInit from "../utils/customInit";

const proximityJSON = {
  type: "boolean_proximity",
  message0: " %1 is %2 %3 %4 from %5",
  args0: [
    {
      type: "input_value",
      name: "LOC1",
      check: types.Location,
    },
    generate_dropdown_comparator("COMP"),
    {
      type: "field_number",
      name: "NUM_BLOCKS",
      value: 1,
      min: 1,
    },
    generate_dropdown_distanceUnits("UNITS"),
    {
      type: "input_value",
      name: "LOC2",
      check: types.Location,
    },
  ],
  inputsInline: true,
  output: types.Boolean,
  style: "logic_blocks",
  tooltip:
    "Given two locations, returns true if the proximity condition between the two is true.",
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_proximity"] = {
  init: function () {
    this.jsonInit(proximityJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_proximity"] = function (block) {
  const number_blocks = block.getFieldValue("NUM_BLOCKS");
  const comparator = block.getFieldValue("COMP");
  const units = block.getFieldValue("UNITS");
  const value_loc1 = Blockly.JavaScript.valueToCode(
    block,
    "LOC1",
    Blockly.JavaScript.ORDER_NONE
  );
  const value_loc2 = Blockly.JavaScript.valueToCode(
    block,
    "LOC2",
    Blockly.JavaScript.ORDER_NONE
  );

  const logicalForm = `{ "condition": {
    "condition_type": "COMPARATOR",
    "input_left" : {"value_extractor": {"distance_between" : [
      { ${value_loc1} },
      { ${value_loc2} }
    ]}},
    "input_right": {
      "value_extractor": ${number_blocks}
    },
    "comparison_type": "${comparator}",
    "comparison_measure": "${units}"
  }
}`;
  return [logicalForm, Blockly.JavaScript.ORDER_NONE];
};
