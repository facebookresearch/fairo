/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Represents a mob in the Minecraft game. Use with accessor block to extract values.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import customInit from "../utils/customInit";
import { generate_dropdown_mobtype } from "../fields/dropdowns";

const mobJSON = {
  type: "value_mob",
  message0: "mob %1",
  args0: [generate_dropdown_mobtype("MOBTYPE")],
  message1: "with name %1",
  args1: [
    {
      type: "field_input",
      name: "NAME",
      value: "",
    },
  ],
  output: types.Mob,
  helpUrl: "",
  colour: 40,
  tooltip:
    "Represents a mob in the Minecraft game. Use with accessor block to extract values.",
  mutator: "labelMutator",
};

Blockly.Blocks["value_mob"] = {
  init: function () {
    this.jsonInit(mobJSON);
    customInit(this);
  },
};

Blockly.JavaScript["value_mob"] = function (block) {
  const mobtype = Blockly.JavaScript.valueToCode(
    block,
    "MOBTYPE",
    Blockly.JavaScript.ORDER_NONE
  );
  const name = Blockly.JavaScript.valueToCode(
    block,
    "NAME",
    Blockly.JavaScript.ORDER_NONE
  );
  const dict = `{ "reference_object": {
    "filters": {
      ${mobtype !== "ANY" && `"has_name": ${mobtype},`}
      ${name && `"has_tag": ${name},`}
    }
  } }`;
  return [dict, Blockly.JavaScript.ORDER_NONE];
};
