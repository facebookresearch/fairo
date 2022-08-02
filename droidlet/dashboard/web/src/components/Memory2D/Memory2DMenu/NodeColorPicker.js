import React, { useEffect, useState, useRef } from "react";
import LensIcon from "@material-ui/icons/Lens"; // Circle
import Grid from "@material-ui/core/Grid";
import ClickAwayListener from "@material-ui/core/ClickAwayListener";

import { BlockPicker } from "react-color";

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {showMenu, onMenuClose, selected_objects, onGroupSubmit, dynamicPositioning, toggleDynamicPositioning, showTriples, toggleShowTriples} props
 *                            showMenu: bool for if menu should be open/close
 *                            onMenuClose: handler to close menu
 *                            selected_objects: dict of all objects selected and their data
 *                            onGroupSubmit: handler to submit grouping
 *                            dynamicPositioning: bool for if map tabular elements should dynamically
 *                                                position themselves based on window position
 *                            toggleDynamicPositioning: handler to toggle DP
 *                            showTriples: bool for if MemoryMapTable should show triples assoc with object
 *                            toggleShowTriples: handler to toggle showTriples
 *                            mapView: the plane which the map is currently displaying
 *                            toggleMapView: handler to change mapView in Memory2D
 *                            centerToBot: handler that centers the stage to the bot
 *                            squareMap: whether the map should fill up the whole pane or limited to square
 *                            toggleSquareMap: handler to change squareMap
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
