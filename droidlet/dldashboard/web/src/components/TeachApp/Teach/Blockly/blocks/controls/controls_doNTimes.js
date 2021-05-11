/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Control flow loop block that loops a given set of commands a given amount of times.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import customInit from "../utils/customInit";
import {
  addCommasBetweenJsonObjects,
  getNumberConcatenatedJsonObjects,
} from "../utils/json";

const doNTimesJSON = {
  type: "controls_doNTimes",
  message0: "repeat %1 times",
  args0: [
    {
      type: "field_number",
      name: "REPEATS",
      value: 1,
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
  tooltip: "Loop a set of commands a given amount of times.",
  mutator: "labelMutator",
};

Blockly.Blocks["controls_doNTimes"] = {
  init: function () {
    this.jsonInit(doNTimesJSON);
    customInit(this);
  },
};

Blockly.JavaScript["controls_doNTimes"] = function (block) {
  const repeat_count = block.getFieldValue("REPEATS");
  let dict = ``;
  let do_statement = Blockly.JavaScript.statementToCode(block, "DO");
  const numChildren = getNumberConcatenatedJsonObjects(do_statement);
  if (numChildren <= 1) {
    const parsedJson = JSON.parse(do_statement);
    parsedJson.control = {
      repeat_key: "FOR",
      repeat_count: repeat_count,
    };
    dict = JSON.stringify(parsedJson);
  } else {
    do_statement = addCommasBetweenJsonObjects(do_statement);
    dict = `{ "action_list": {
    "list": [${do_statement}],
    "control": {
      "repeat_key": "FOR",
      "repeat_count": "${repeat_count}"
    }
  }}`;
  }
  return dict;
};
