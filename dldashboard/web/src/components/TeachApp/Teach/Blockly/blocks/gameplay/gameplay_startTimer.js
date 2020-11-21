/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Statement block that starts a timer. Its value can be accessed with the Timer block.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";

const timerJSON = {
  type: "gameplay_startTimer",
  message0: "Start Timer",
  previousStatement: null,
  nextStatement: null,
  helpUrl: "",
  colour: 20,
  tooltip: 'Starts a timer. Use "Timer" block to access current time.',
  mutator: "labelMutator",
};

Blockly.Blocks["gameplay_startTimer"] = {
  init: function () {
    this.jsonInit(timerJSON);
  },
};

Blockly.JavaScript["gameplay_startTimer"] = function (block) {
  return ["", Blockly.JavaScript.ORDER_NONE];
};
