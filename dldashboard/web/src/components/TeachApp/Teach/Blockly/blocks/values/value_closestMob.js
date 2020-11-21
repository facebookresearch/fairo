/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Represents the closest mob to the given position in the Minecraft game. Use with accessor block to extract values.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";
import { generate_dropdown_mobtype } from "../fields/dropdowns";

const mobJSON = {
  type: "value_closestMob",
  message0: "closest %1 to %2",
  args0: [
    generate_dropdown_mobtype("MOBTYPE"),
    {
      type: "input_value",
      name: "LOC",
      check: [types.Location],
    },
  ],
  output: types.Mob,
  inputsInline: true,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Represents the closest mob to the given position in the Minecraft game. Use with accessor block to extract values.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_closestMob"] = {
  init: function () {
    this.jsonInit(mobJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_closestMob"] = function (block) {
  const mobtype = block.getFieldValue("MOBTYPE");
  const location = Blockly.JavaScript.valueToCode(
    block,
    "LOC",
    Blockly.JavaScript.ORDER_NONE
  );
  const dict = `{ "reference_object": {
    "filters": {
      "has_name": "${mobtype}",
      "argmin" : {
        "ordinal" : "FIRST",
        "quantity" : {
          "linear_extent" : {
            ${location}
          }
        }
      }
    }
  } }`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
