/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Statement block that stops the current timer. Access its value with the Timer block.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";

const timerJSON = {
  type: "gameplay_stopTimer",
  message0: "Stop Timer",
  previousStatement: null,
  nextStatement: null,
  helpUrl: "",
  colour: 20,
  tooltip:
    'Stops the current timer without resetting it. Use "Timer" block to access current time.',
  mutator: "labelMutator",
};

Blockly.Blocks["gameplay_stopTimer"] = {
  init: function () {
    this.jsonInit(timerJSON);
  },
};

Blockly.JavaScript["gameplay_stopTimer"] = function (block) {
  return ["", Blockly.JavaScript.ORDER_NONE];
};
