/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Control flow loop block that loops a given set of commands forever.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import customInit from "../utils/customInit";
import {
  addCommasBetweenJsonObjects,
  getNumberConcatenatedJsonObjects,
} from "../utils/json";

const doUntilJSON = {
  type: "controls_doForever",
  message0: "do forever %1",
  args0: [
    {
      type: "input_statement",
      name: "DO",
    },
  ],
  previousStatement: null,
  nextStatement: null,
  style: "loop_blocks",
  helpUrl: "",
  tooltip: "Repeatedly loop a set of commands.",
  mutator: "labelMutator",
};

Blockly.Blocks["controls_doForever"] = {
  init: function () {
    this.jsonInit(doUntilJSON);
    customInit(this);
  },
};

Blockly.JavaScript["controls_doForever"] = function (block) {
  let dict = ``;
  let do_statement = Blockly.JavaScript.statementToCode(block, "DO");
  const numChildren = getNumberConcatenatedJsonObjects(do_statement);
  if (numChildren <= 1) {
    const parsedJson = JSON.parse(do_statement);
    parsedJson.control = {
      repeat_key: "FOR",
      repeat_count: "forever",
    };
    dict = JSON.stringify(parsedJson);
  } else {
    do_statement = addCommasBetweenJsonObjects(do_statement);
    dict = `{ "action_list": {
    "list": [${do_statement}],
    "control": {
      "repeat_key": "FOR",
      "repeat_count": "forever"
    }
  }}`;
  }
  return dict;
};
