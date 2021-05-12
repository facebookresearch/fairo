/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block that represents an event happening in-game (is the event happening or not).
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const timeEventJSON = {
  type: "boolean_timeEvent",
  message0: "game event %1",
  args0: [
    {
      type: "field_dropdown",
      name: "EVENT",
      options: [
        ["sunset", "SUNSET"],
        ["sunrise", "SUNRISE"],
        ["day", "DAY"],
        ["night", "NIGHT"],
        ["rain", "RAIN"],
        ["sunny", "SUNNY"],
      ],
    },
  ],
  inputsInline: true,
  output: types.Boolean,
  tooltip:
    "Represents an in-game event happening. Returns true when that event is happening.",
  style: "logic_blocks",
  mutator: "labelMutator",
};

Blockly.Blocks["boolean_timeEvent"] = {
  init: function () {
    this.jsonInit(timeEventJSON);
    customInit(this);
  },
};

Blockly.JavaScript["boolean_timeEvent"] = function (block) {
  const event = block.getFieldValue("EVENT");
  const dict = `{
    "condition": {
      "condition_type": "TIME",
      "special_time_event": "${event}"
    }
  }`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
