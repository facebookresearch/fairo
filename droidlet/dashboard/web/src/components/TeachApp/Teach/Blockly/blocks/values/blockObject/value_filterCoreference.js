/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Filter option for filtering by object containing a coreference.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../../utils/types";
import customInit from "../../utils/customInit";

const filterCoreferenceJSON = {
  type: "value_filterCoreference",
  message0: "contains coreference",
  args0: [],
  inputsInline: true,
  previousStatement: types.Filter,
  nextStatement: types.Filter,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Filter option for filtering by whether or not the object has a coreference. Use with block object block.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_filterCoreference"] = {
  init: function () {
    this.jsonInit(filterCoreferenceJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_filterCoreference"] = function (block) {
  const dict = `"contains_coreference": "yes",\n`;
  return dict;
};
