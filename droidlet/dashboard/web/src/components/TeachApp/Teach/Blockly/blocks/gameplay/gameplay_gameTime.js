/* Copyright (c) Facebook, Inc. and its affiliates. */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";

const clockJSON = {
  type: "gameplay_gameTime",
  message0: "Game time %1",
  args0: [
    {
      type: "field_dropdown",
      name: "TIME",
      options: [
        ["sunrise", "SUNRISE"],
        ["day", "DAY"],
        ["noon", "NOON"],
        ["sunset", "SUNSET"],
        ["night", "NIGHT"],
        ["midnight", "MIDNIGHT"],
      ],
    },
  ],
  output: types.Time,
  helpUrl: "",
  colour: 20,
  tooltip: "Gets the time of day. Use with accessor or compare blocks.",
  mutator: "labelMutator",
};

Blockly.Blocks["gameplay_gameTime"] = {
  init: function () {
    this.jsonInit(clockJSON);
  },
};

Blockly.JavaScript["gameplay_gameTime"] = function (block) {
  return ["", Blockly.JavaScript.ORDER_NONE];
};
