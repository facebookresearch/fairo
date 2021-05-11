/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Entry point for loading all the custom block code.
 */

import "./locations/location";
import "./locations/location_relative";
import "./locations/player_location";
import "./locations/agent_location";

import "./actions/agent_move";

import "./controls/controls_doUntil";
import "./controls/controls_doForever";
import "./controls/controls_doNTimes";
import "./controls/controls_custom_ifelse";
import "./controls/controls_command";
import "./controls/controls_savedCommand";

import "./booleans/boolean_proximity";
import "./booleans/boolean_comparator";
import "./booleans/boolean_stringMatches";
import "./booleans/boolean_and";
import "./booleans/boolean_or";
import "./booleans/boolean_not";
import "./booleans/boolean_timeEvent";
import "./booleans/boolean_timePassed";

import "./values/value_number";
import "./values/value_text";
import "./values/value_accessor";
import "./values/value_mob";
import "./values/value_closestMob";
import "./values/blockObject/value_blockObject";
import "./values/blockObject/value_filterColor";
import "./values/blockObject/value_filterLocation";
import "./values/blockObject/value_filterName";
import "./values/blockObject/value_filterSize";
import "./values/blockObject/value_filterCoreference";

import "./gameplay/gameplay_clock";
import "./gameplay/gameplay_gameTime";
import "./gameplay/gameplay_timer";
import "./gameplay/gameplay_startTimer";
import "./gameplay/gameplay_stopTimer";
import "./gameplay/gameplay_resetTimer";

// not blocks
import "./utils/modifyUpdateCollapsed"; // modify collapse behavior
import "./utils/labelMutator"; // save all labels to database
