/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block representing the agent's location.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";

const agentLocationJSON = {
  type: "agent_location",
  message0: "Agent Location",
  output: types.Location,
  colour: 230,
  tooltip: "Location in game world of the Craftassist bot.", // TODO maybe generalize for locobot?
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["agent_location"] = {
  init: function () {
    this.jsonInit(agentLocationJSON);
  },
};

Blockly.JavaScript["agent_location"] = function (block) {
  const logicalForm = `"location": { "reference_object": { "special_reference": "AGENT" } }`;
  return [logicalForm, Blockly.JavaScript.ORDER_NONE];
};
