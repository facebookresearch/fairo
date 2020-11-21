/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Dropdown fields for use with Blockly's custom block system.
 */

const UNITS_DISTANCE = ["blocks", "meters", "feet"];

const UNITS_TIME = ["seconds", "minutes", "hours", "days", "ticks"];

// Given a field name, return the dropdown field with that name.
export const generate_dropdown_distanceUnits = (name) => ({
  type: "field_dropdown",
  name: name,
  options: UNITS_DISTANCE.map((unit) => [unit, unit.toUpperCase()]),
});

// Given a field name, return the dropdown field with that name.
export const generate_dropdown_comparator = (name) => ({
  type: "field_dropdown",
  name: name,
  options: [
    [">", "GREATER_THAN"],
    ["<", "LESS_THAN"],
    ["≥", "GREATER_THAN_EQUAL"],
    ["≤", "LESS_THAN_EQUAL"],
    ["=", "EQUAL"],
    ["≠", "NOT_EQUAL"],
  ],
});

export const generate_dropdown_mobtype = (name) => ({
  type: "field_dropdown",
  name: name,
  options: [
    ["Any", "ANY"],
    ["Creeper", "CREEPER"],
    ["Skeleton", "SKELETON"],
    ["Spider", "SPIDER"],
    ["Giant", "GIANT"],
    ["Zombie", "ZOMBIE"],
    ["Slime", "SLIME"],
    ["Ghast", "GHAST"],
    ["Pig Zombie", "PIG_ZOMBIE"],
    ["Enderman", "ENDERMAN"],
    ["Cave Spider", "CAVE_SPIDER"],
    ["Silverfish", "SILVERFISH"],
    ["Blaze", "BLAZE"],
    ["Lava Slime", "LAVA_SLIME"],
    ["Ender Dragon", "ENDER_DRAGON"],
    ["Wither Boss", "WITHER_BOSS"],
    ["Bat", "BAT"],
    ["Witch", "WITCH"],
    ["Guardian", "GUARDIAN"],
    ["Pig", "PIG"],
    ["Sheep", "SHEEP"],
    ["Cow", "COW"],
    ["Chicken", "CHICKEN"],
    ["Squid", "SQUID"],
    ["Wolf", "WOLF"],
    ["Mushroom Cow", "MUSHROOM_COW"],
    ["Snow Man", "SNOW_MAN"],
    ["Ozelot", "OZELOT"],
    ["Villager Golem", "VILLAGER_GOLEM"],
    ["Entity Horse", "ENTITY_HORSE"],
    ["Rabbit", "RABBIT"],
    ["Villager", "VILLAGER"],
  ],
});

export const generate_dropdown_timeUnits = (name) => ({
  type: "field_dropdown",
  name: name,
  options: UNITS_TIME.map((unit) => [unit, unit.toUpperCase()]),
});
