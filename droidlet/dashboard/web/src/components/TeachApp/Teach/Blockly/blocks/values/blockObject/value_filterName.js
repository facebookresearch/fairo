/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Filter option for filtering by reference object name.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../../utils/types";
import customInit from "../../utils/customInit";

const filterNameJSON = {
  type: "value_filterName",
  message0: "has name %1",
  args0: [
    {
      type: "field_input",
      name: "NAME",
    },
  ],
  inputsInline: true,
  previousStatement: types.Filter,
  nextStatement: types.Filter,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Filter option for filtering by reference object name. Use with block object block.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_filterName"] = {
  init: function () {
    this.jsonInit(filterNameJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_filterName"] = function (block) {
  const name = block.getFieldValue("NAME");
  const dict = `"has_name": "${name}",\n`;
  return dict;
};
