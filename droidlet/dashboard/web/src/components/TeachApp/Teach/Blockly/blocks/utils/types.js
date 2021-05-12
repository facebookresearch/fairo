/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Types as strings for use with Blockly's typing system.
 */

const types = {
  Location: "Location",
  Boolean: "Boolean",
  Number: "Number",
  String: "String",
  CommandDict: "CommandDict",
  Time: "Time",
  Mob: "Mob",
  BlockObject: "BlockObject",
  Filter: "Filter",
};

// Maps types to the fields we can "access" from those types of blocks.
export const TYPES_TO_FIELD_OPTIONS = {
  [types.Location]: [
    ["x", "X"],
    ["y", "Y"],
    ["z", "Z"],
    ["pitch", "PITCH"],
    ["yaw", "YAW"],
  ],
  [types.Time]: [
    ["time (ticks)", "TICKS"],
    ["time (seconds)", "SECONDS"],
    ["time (minutes)", "MINUTES"],
    ["time (hours)", "HOURS"],
    ["time (days)", "DAYS"],
    ["time string (hh:mm:ss)", "TIME"],
  ],
  [types.Mob]: [
    ["location", "LOC"],
    ["location X", "X"],
    ["location Y", "Y"],
    ["location Z", "Z"],
    ["type", "TYPE"],
    ["name", "NAME"],
  ],
  [types.BlockObject]: [
    ["location", "LOC"],
    ["location X", "X"],
    ["location Y", "Y"],
    ["location Z", "Z"],
    ["name", "NAME"],
    ["size", "SIZE"],
    ["color", "COLOR"],
  ],
};

export default types;
