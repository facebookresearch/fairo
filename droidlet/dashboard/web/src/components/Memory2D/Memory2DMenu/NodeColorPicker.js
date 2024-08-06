/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D/NodeColorPicker.js

import React, { useEffect, useState, useRef } from "react";
import LensIcon from "@material-ui/icons/Lens"; // Circle
import Grid from "@material-ui/core/Grid";
import ClickAwayListener from "@material-ui/core/ClickAwayListener";

import { BlockPicker } from "react-color";

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {type, count, color, setNodeColoring} props
 *                            type: the nodeType (i.e. LocationNode, MobNode, etc.)
 *                            count: number of this type of node currently plotted on map
 *                            color: currently set color for this type of node
 *                            setNodeColoring: handler to change color of this type of node
 */
export default function NodeColorPicker(props) {
  const [nodeColor, setNodeColor] = useState(props.color);
  const [pickerVisible, setPickerVisible] = useState(false);

  const pickerEl = useRef(null);
  const circEl = useRef(null);

  const pickerWidth = 170;

  // Positions color picker at circle
  useEffect(() => {
    if (pickerEl.current) {
      let { x, y, height, width } = circEl.current.getBoundingClientRect();
      pickerEl.current.style.position = "fixed";
      pickerEl.current.style.left = x + width / 2 - pickerWidth / 2 + "px";
      pickerEl.current.style.top = y + height / 2 + 14 + "px"; // 14 added for triangle
      pickerEl.current.style.zindex = 10;
    }
  }, [pickerVisible]);

  // Updates circle color
  useEffect(() => {
    circEl.current.style.color = nodeColor;
  }, [nodeColor]);

  return (
    <div>
      <Grid container spacing={2}>
        <Grid item>
          <LensIcon
            ref={circEl}
            onClick={() => {
              setPickerVisible((prev) => {
                return !prev;
              });
            }}
          />
        </Grid>
        <Grid item>
          {props.type} ({props.count})
        </Grid>
      </Grid>
      {pickerVisible && (
        <ClickAwayListener
          onClickAway={() => {
            setPickerVisible(false);
          }}
        >
          <div ref={pickerEl}>
            <BlockPicker
              color={nodeColor}
              colors={DEFAULT_COLOR_CHOICES}
              width={pickerWidth}
              triangle="top"
              onChange={({ hex: color }) => {
                color = color.toUpperCase(); // react-color likes lowercase, Konva disagrees
                setNodeColor(color);
                props.setNodeColoring(color);
              }}
            />
          </div>
        </ClickAwayListener>
      )}
    </div>
  );
}

const DEFAULT_COLOR_CHOICES = [
  "#F47373",
  "#DCE775",
  "#2CCCE4",
  "#37D67A",
  "#0000FF",
  "#D9E3F0",
  "#697689",
  "#555555",
  "#FF8A65",
  "#BA68C8",
];
