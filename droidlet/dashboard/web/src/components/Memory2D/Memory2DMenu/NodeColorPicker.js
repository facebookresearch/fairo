import React, { useEffect, useState, useRef, PureComponent } from "react";
import ButtonGroup from "@material-ui/core/ButtonGroup";
import Button from "@material-ui/core/Button";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import HelpIcon from "@material-ui/icons/Help";
import Tooltip from "@material-ui/core/Tooltip";
import Drawer from "@material-ui/core/Drawer";
import Divider from "@material-ui/core/Divider";
import TextField from "@material-ui/core/TextField";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Checkbox from "@material-ui/core/Checkbox";
import FiberManualRecordIcon from "@material-ui/icons/FiberManualRecord"; // Circle
import Grid from "@material-ui/core/Grid";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";

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
  const [nodeColor, setNodeColor] = React.useState({ hex: "#FF6900" });

  const pickerEl = useRef(null);
  const circEl = useRef(null);

  useEffect(() => {
    console.log(circEl.current.getBoundingClientRect());
    let circStyle = circEl.current.style;
    let divStyle = pickerEl.current.style;
    // let { x, y, height, width } = circEl.current.getBoundingClientRect();
    // console.log(height, width, x, y);
    pickerEl.current.style.position = "absolute";
    pickerEl.current.style.left = 0;
    pickerEl.current.style.bottom = "100px";
    console.log(circStyle.position, circStyle.left, circStyle.bottom);
    console.log(divStyle.position, divStyle.left, divStyle.bottom);
  });

  useEffect(() => {
    circEl.current.style.color = nodeColor.hex;
  }, [nodeColor]);

  return (
    <div>
      <div ref={pickerEl}>
        <BlockPicker
          color={nodeColor}
          width={170}
          triangle="top"
          onChange={(color) => {
            setNodeColor(color);
          }}
        />
      </div>
      LocationNode (2)
      <FiberManualRecordIcon ref={circEl} />
    </div>
  );
}
