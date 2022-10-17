/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D/Memory2DConstants.js

import React from "react";

import Box from "@material-ui/core/Box";
import Tooltip from "@material-ui/core/Tooltip";

/*#########################
####  Stage Constants  ####
#########################*/

// Defaults

export const INITIAL_HEIGHT = 400;
export const INITIAL_WIDTH = 600;

export const SCALE_FACTOR = 1.2;

export const DEFAULT_SPACING = 12;

export const DEFAULT_NODE_COLORINGS = {
  PlayerNode: "#F47373",
  AttentionNode: "#DCE775",
  LocationNode: "#2CCCE4",
  MobNode: "#37D67A",
  ItemStackNode: "#0000FF",
};

// Plotting Customization

// Radii
export const DETECTION_FROM_MEMORY_RADIUS = 6;
export const OBSTACLE_MAP_RADIUS = 2;
export const CLUSTER_RADIUS = 6;
export const BOT_RADIUS = 10;

// Axis
export const ROOT_POS = [40, 25]; // pixel position from bottom left of stage
export const AXES_MARGIN = 20;

// Notches
export const NOTCH_SPACING = [30, 25]; // every 30 pixels for horz axis, 25 for vert
export const NOTCH_LENGTH = 6;
export const HORZ_NOTCH_TEXT_OFFSET = [10, 15]; // 10 left, 15 up
export const VERT_NOTCH_TEXT_OFFSET = [32, 5]; // 35 left, 5 up

/*#################################
####  Miscellaneous Constants  ####
#################################*/

export const FONT = "Segoe UI";

export const MENU_WIDTH = 450;

/**
 *
 * @param {entry into table} e
 * @param {where to place tooltip} placement
 * @returns If a table entry is too long to fit properly,
 *          shorten it and allow user to see full entry on hover
 */
export function shortenLongTableEntries(
  e,
  cutoff = 12,
  placement = "left-start"
) {
  // table entries currently at fontsize 14
  // ex. string of length 12 will be approximately 96px
  //     so we keep first 4, ellipsis, and last 4 characters
  //     to ensure entire string < 100px
  let numCharsToKeep = (cutoff - 4) >> 1;
  if (e && e.length > cutoff) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        maxHeight={20}
      >
        <Tooltip title={e} placement={placement} interactive leaveDelay={500}>
          {/* keep first 4 and last 4 characters to ensure entire string < 100px */}
          <p>
            {e.substring(0, numCharsToKeep) +
              "..." +
              e.substring(e.length - numCharsToKeep)}
          </p>
        </Tooltip>
      </Box>
    );
  }
  return e;
}
