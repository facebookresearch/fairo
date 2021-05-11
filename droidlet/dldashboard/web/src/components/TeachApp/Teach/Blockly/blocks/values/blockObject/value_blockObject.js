/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Represents a group of blocks in the world. Use with accessor block to extract values.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../../utils/types";
import customInit from "../../utils/customInit";

const blockObjectJSON = {
  type: "value_blockObject",
  message0: "block object",
  message1: "filters %1",
  args1: [
    {
      type: "input_statement",
      name: "FILTERS",
      check: [types.Filter],
    },
  ],
  output: types.BlockObject,
  inputsInline: true,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Represents a group of blocks in the world. Use with accessor block to extract values.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_blockObject"] = {
  init: function () {
    this.jsonInit(blockObjectJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_blockObject"] = function (block) {
  const filters = Blockly.JavaScript.statementToCode(block, "FILTERS");
  const dict = `{ "reference_object": {
    "filters": {
      ${filters.slice(0, -2) /* remove trailing comma to fit JSON spec */}
    }
  }}`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
