/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Blockly toolbox (sidebar) specification, specified using our Blockly/React
 * interface components.
 */

import React from "react";
import { Block, Category } from "./index";

const Toolbox = () => (
  <>
    <Category name="Booleans">
      <Block type="boolean_proximity" />
      <Block type="boolean_timeEvent" />
      <Block type="boolean_timePassed" />
      <Block type="boolean_comparator" />
      <Block type="boolean_stringMatches" />
      <Block type="boolean_and" />
      <Block type="boolean_or" />
      <Block type="boolean_not" />
    </Category>
    <Category name="Control">
      <Block type="controls_custom_ifelse" />
      <Block type="controls_doUntil" />
      <Block type="controls_doForever" />
      <Block type="controls_doNTimes" />
    </Category>
    <Category name="Location">
      <Block type="location" />
      <Block type="location_relative" />
      <Block type="player_location" />
      <Block type="agent_location" />
    </Category>
    {/* timer blocks disabled for now, while they're out of the grammar spec
    <Category name="Gameplay">
      <Block type="gameplay_clock" />
      <Block type="gameplay_gameTime" />
      <Block type="gameplay_timer" />
      <Block type="gameplay_startTimer" />
      <Block type="gameplay_stopTimer" />
      <Block type="gameplay_resetTimer" />
    </Category> */}
    <Category name="Actions">
      <Block type="agent_move" />
    </Category>
    <Category name="Values">
      <Block type="value_number" />
      <Block type="value_text" />
      <Block type="value_accessor" />
      <Block type="value_mob" />
      <Block type="value_closestMob" />
      <Category name="Block Object">
        <Block type="value_blockObject" />
        <Block type="value_filterColor" />
        <Block type="value_filterLocation" />
        <Block type="value_filterName" />
        <Block type="value_filterSize" />
        <Block type="value_filterCoreference" />
      </Category>
    </Category>
  </>
);

export default Toolbox;
