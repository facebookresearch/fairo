/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Statement block that resets the current timer.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";

const timerJSON = {
  type: "gameplay_resetTimer",
  message0: "Reset Timer",
  previousStatement: null,
  nextStatement: null,
  helpUrl: "",
  colour: 20,
  tooltip:
    'Resets and stops the current timer. Use "Timer" block to access current time.',
  mutator: "labelMutator",
};

Blockly.Blocks["gameplay_resetTimer"] = {
  init: function () {
    this.jsonInit(timerJSON);
  },
};

Blockly.JavaScript["gameplay_resetTimer"] = function (block) {
  return ["", Blockly.JavaScript.ORDER_NONE];
};
