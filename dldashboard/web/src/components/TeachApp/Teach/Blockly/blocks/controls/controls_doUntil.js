/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Control flow loop block that loops a given set of commands
 * until a certain boolean condition is true.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";
import {
  addCommasBetweenJsonObjects,
  getNumberConcatenatedJsonObjects,
} from "../utils/json";

const doUntilJSON = {
  type: "controls_doUntil",
  message0: "repeat until %1",
  args0: [
    {
      type: "input_value",
      name: "BOOL",
      check: types.Boolean,
    },
  ],
  message1: "do %1",
  args1: [
    {
      type: "input_statement",
      name: "DO",
    },
  ],
  previousStatement: null,
  nextStatement: null,
  style: "loop_blocks",
  helpUrl: "",
  tooltip:
    "Until a boolean condition is met, repeatedly loop a set of commands.",
  mutator: "labelMutator",
};

Blockly.Blocks["controls_doUntil"] = {
  init: function () {
    this.jsonInit(doUntilJSON);
    customInit(this);
  },
};

Blockly.JavaScript["controls_doUntil"] = function (block) {
  const condition = Blockly.JavaScript.valueToCode(
    block,
    "BOOL",
    Blockly.JavaScript.ORDER_NONE
  );
  let do_statement = Blockly.JavaScript.statementToCode(block, "DO");
  let dict = ``;
  const numChildren = getNumberConcatenatedJsonObjects(do_statement);
  if (numChildren <= 1) {
    const parsedJson = JSON.parse(do_statement);
    parsedJson.control = {
      stop_condition: JSON.parse(condition),
    };
    dict = JSON.stringify(parsedJson);
  } else {
    do_statement = addCommasBetweenJsonObjects(do_statement);
    dict = `{ "action_list": { "list": [${do_statement}],
    "control": {
      "stop_condition": ${condition}
    }
  }}`;
  }
  return dict;
};
