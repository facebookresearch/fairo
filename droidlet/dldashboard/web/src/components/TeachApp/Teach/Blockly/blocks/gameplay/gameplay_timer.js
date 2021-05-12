/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block allowing access to the current timer. Timer controls are the
 * Start Timer, Stop Timer, and Reset Timer blocks.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";

const clockJSON = {
  type: "gameplay_timer",
  message0: "Timer",
  output: types.Time,
  helpUrl: "",
  colour: 20,
  tooltip:
    'Gets current value of timer. Use with accessor or compare blocks. Set timer with "Start Timer" block.',
  mutator: "labelMutator",
};

Blockly.Blocks["gameplay_timer"] = {
  init: function () {
    this.jsonInit(clockJSON);
  },
};

Blockly.JavaScript["gameplay_timer"] = function (block) {
  return ["", Blockly.JavaScript.ORDER_NONE];
};
