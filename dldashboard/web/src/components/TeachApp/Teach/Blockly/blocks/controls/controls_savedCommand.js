/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Represents the overall structure of a command. Used for commands with a series of steps.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import customInit from "../utils/customInit";

const commandJSON = {
  type: "controls_savedCommand",
  message0: "command %1 %2",
  args0: [
    {
      type: "input_dummy",
    },
    {
      type: "input_statement",
      name: "STEPS",
    },
  ],
  colour: 230,
  previousStatement: null,
  nextStatement: null,
  tooltip: "Place your command steps in this block.",
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["controls_savedCommand"] = {
  init: function () {
    this.jsonInit(commandJSON);
    customInit(this);
  },
};

Blockly.JavaScript["controls_savedCommand"] = function (block) {
  return Blockly.JavaScript.statementToCode(block, "STEPS");
};
