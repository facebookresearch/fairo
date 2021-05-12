/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Filter option for filtering by reference object size.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../../utils/types";
import customInit from "../../utils/customInit";

const filterSizeJSON = {
  type: "value_filterSize",
  message0: "has size %1",
  args0: [
    {
      type: "field_input",
      name: "SIZE",
      check: [types.String],
    },
  ],
  inputsInline: true,
  previousStatement: types.Filter,
  nextStatement: types.Filter,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Filter option for filtering by reference object size. Use with block object block.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_filterSize"] = {
  init: function () {
    this.jsonInit(filterSizeJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_filterSize"] = function (block) {
  const size = block.getFieldValue("SIZE");
  const dict = `"has_size": "${size}",\n`;
  return dict;
};
