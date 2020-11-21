/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Control flow loop block that runs one set of commands if a condition is true, otherwise
 * another set of commands.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";
import {
  addCommasBetweenJsonObjects,
  getNumberConcatenatedJsonObjects,
} from "../utils/json";

const ifElseJSON = {
  type: "controls_custom_ifelse",
  message0: "if %1",
  args0: [
    {
      type: "input_value",
      name: "IF",
      check: types.Boolean,
    },
  ],
  message1: "then %1",
  args1: [
    {
      type: "input_statement",
      name: "DO",
    },
  ],
  previousStatement: null,
  nextStatement: null,
  style: "logic_blocks",
  tooltip: "%{BKYCONTROLS_IF_TOOLTIP_2}",
  helpUrl: "%{BKY_CONTROLS_IF_HELPURL}",
  extensions: ["controls_if_tooltip"],
  mutator: "labelMutator",
};

Blockly.Blocks["controls_custom_ifelse"] = {
  init: function () {
    this.jsonInit(ifElseJSON);
    customInit(this);
  },
};

Blockly.JavaScript["controls_custom_ifelse"] = function (block) {
  const condition = Blockly.JavaScript.valueToCode(
    block,
    "IF",
    Blockly.JavaScript.ORDER_NONE
  );
  let do_statement = Blockly.JavaScript.statementToCode(block, "DO");
  let dict = ``;
  const numChildren = getNumberConcatenatedJsonObjects(do_statement);
  if (numChildren <= 1) {
    const parsedJson = JSON.parse(do_statement);
    parsedJson.control = {
      on_condition: JSON.parse(condition),
    };
    dict = JSON.stringify(parsedJson);
  } else {
    do_statement = addCommasBetweenJsonObjects(do_statement);
    dict = `{ "action_list": { "list": [${do_statement}],
    "control": {
      "on_condition": ${condition}
    }
  }}`;
  }
  return dict;
};
