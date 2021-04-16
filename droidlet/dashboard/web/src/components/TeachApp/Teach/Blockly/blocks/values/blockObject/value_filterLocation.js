/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Filter option for filtering by reference object location.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../../utils/types";
import customInit from "../../utils/customInit";

const filterLocationJSON = {
  type: "value_filterLocation",
  message0: "at location %1",
  args0: [
    {
      type: "input_value",
      name: "LOC",
      check: [types.Location],
    },
  ],
  inputsInline: true,
  previousStatement: types.Filter,
  nextStatement: types.Filter,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Filter option for filtering by reference object location. Use with block object block.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_filterLocation"] = {
  init: function () {
    this.jsonInit(filterLocationJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_filterLocation"] = function (block) {
  const location = Blockly.JavaScript.valueToCode(
    block,
    "LOC",
    Blockly.JavaScript.ORDER_NONE
  );
  const dict = `${location},\n`;
  return dict;
};
