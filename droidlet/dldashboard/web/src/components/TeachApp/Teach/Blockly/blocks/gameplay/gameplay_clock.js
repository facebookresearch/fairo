/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Block representing current value of the in-game clock. Access value using
 * the accessor block in the values category.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";

const clockJSON = {
  type: "gameplay_clock",
  message0: "Clock",
  output: types.Time,
  helpUrl: "",
  colour: 20,
  tooltip: "Gets the time of day. Use with accessor or compare blocks.",
  mutator: "labelMutator",
};

Blockly.Blocks["gameplay_clock"] = {
  init: function () {
    this.jsonInit(clockJSON);
  },
};

Blockly.JavaScript["gameplay_clock"] = function (block) {
  return ["", Blockly.JavaScript.ORDER_NONE];
};
