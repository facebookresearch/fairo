/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Value block representing a location relative to the agent.
 */

import * as Blockly from "blockly/core";
import "blockly/javascript";
import types from "../utils/types";
import { generate_dropdown_distanceUnits } from "../fields/dropdowns";
import customInit from "../utils/customInit";

const locationRelativeJSON = {
  type: "location_relative",
  message0: "%1 %2 in direction %3 relative to %4",
  args0: [
    {
      type: "field_number",
      name: "count",
      value: 1,
      min: 0,
    },
    generate_dropdown_distanceUnits("units"),
    {
      type: "field_dropdown",
      name: "direction",
      options: [
        ["left", "LEFT"],
        ["right", "RIGHT"],
        ["up", "UP"],
        ["down", "DOWN"],
        ["front", "FRONT"],
        ["back", "BACK"],
      ],
    },
    {
      type: "field_dropdown",
      name: "person",
      options: [
        ["agent", "AGENT"],
        ["speaker", "SPEAKER"],
      ],
    },
  ],
  output: types.Location,
  colour: 230,
  tooltip:
    "A relative location, specifying a location relative to the agent or speaker.",
  helpUrl: "",
  mutator: "labelMutator",
};

Blockly.Blocks["location_relative"] = {
  init: function () {
    this.jsonInit(locationRelativeJSON);
    customInit(this);
  },
};

Blockly.JavaScript["location_relative"] = function (block) {
  var number_count = block.getFieldValue("count");
  var dropdown_units = block.getFieldValue("units");
  var dropdown_direction = block.getFieldValue("direction");
  const logicalForm = `"location": {
  "steps": ${number_count},
  "has_measure": "${dropdown_units}",
  "relative_direction": "${dropdown_direction}"
}`;

  return [logicalForm, Blockly.JavaScript.ORDER_NONE];
};
