/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Statement block that moves the agent towards a given location.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";

const agentMoveJSON = {
  type: "agent_move",
  message0: "move agent towards %1",
  args0: [
    {
      type: "input_value",
      name: "pos",
      check: types.Location,
    },
  ],
  previousStatement: types.CommandDict,
  nextStatement: types.CommandDict,
  style: "loop_blocks",
  helpUrl: "",
  tooltip: "Move the agent towards a location.",
  mutator: "labelMutator",
};

Blockly.Blocks["agent_move"] = {
  init: function () {
    this.jsonInit(agentMoveJSON);
    customInit(this);
  },
};

Blockly.JavaScript["agent_move"] = function (block) {
  const pos = Blockly.JavaScript.valueToCode(
    block,
    "pos",
    Blockly.JavaScript.ORDER_NONE
  ); // order_none avoids any extra parentheses
  const logicalForm = `{
  "action_type": "move",
  ${pos}
}`;

  return logicalForm;
};
