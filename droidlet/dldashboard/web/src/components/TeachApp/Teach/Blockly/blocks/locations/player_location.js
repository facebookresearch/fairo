/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block representing the location of the player (or speaker).
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";

const playerLocationJSON = {
  type: "player_location",
  message0: "Player Location",
  output: types.Location,
  colour: 230,
  tooltip: "Location in game world of your Minecraft player.",
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["player_location"] = {
  init: function () {
    this.jsonInit(playerLocationJSON);
  },
};

Blockly.JavaScript["player_location"] = function (block) {
  const logicalForm = `"location": { "reference_object": { "special_reference": "SPEAKER" } }`;
  return [logicalForm, Blockly.JavaScript.ORDER_NONE];
};
