/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * A basic block representing a text message or string.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const numberJSON = {
  type: "value_textj",
  message0: "%1",
  args0: [
    {
      type: "field_input",
      name: "TEXT",
      value: "",
    },
  ],
  output: types.String,
  style: "math_blocks",
  extensions: ["parent_tooltip_when_inline"],
  mutator: "labelMutator",
};

Blockly.Blocks["value_text"] = {
  init: function () {
    this.jsonInit(numberJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_text"] = function (block) {
  const input = block.getFieldValue("TEXT");
  return [`"${input}"`, Blockly.JavaScript.ORDER_NONE];
};
