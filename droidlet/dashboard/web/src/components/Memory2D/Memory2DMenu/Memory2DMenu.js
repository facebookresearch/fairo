/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D/Memory2DMenu.js

import React from "react";

import * as M2DC from "../Memory2DConstants";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
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
import Grid from "@material-ui/core/Grid";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import NodeColorPicker from "./NodeColorPicker";
import Accordion from "@material-ui/core/Accordion";
import AccordionSummary from "@material-ui/core/AccordionSummary";
import AccordionDetails from "@material-ui/core/AccordionDetails";
import ExpandMoreIcon from "@material-ui/icons/ExpandMore";

const menuTheme = createMuiTheme({
  palette: {
    type: "dark",
  },
});

const groupingHelpText =
  "Group selected objects and store as 'is_a' predicate triple in memory. To select objects, use Cmd+Click or right click to start describing a selection window";

const dynamicPosHelpText =
  "Enabling this makes it so any tabular component on the map automatically repositions itself to remain on screen when dragging. May reduce performance.";

const nodeDetailsHelpText =
  "List of all node types on map. Click on colored circle to change color. Click away from color picker menu to close it.";

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {showMenu, onMenuClose, selected_objects, onGroupSubmit, dynamicPositioning,
 *            toggleDynamicPositioning, showTriples, toggleShowTriples, nodeTypeInfo, setNodeColoring} props
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
 *                            nodeTypeInfo: object with count + color information for each nodeType
 *                            setNodeColoring: handler to set color of node type
 */
export default function Memory2DMenu(props) {
  const [groupName, setGroupName] = React.useState("");
  const [groupingCount, setGroupingCount] = React.useState(0);
  const [mapView, setMapView] = React.useState(props.mapView);

  return (
    <ThemeProvider theme={menuTheme}>
      <Drawer anchor="right" open={props.showMenu} onClose={props.onMenuClose}>
        <div style={{ width: M2DC.MENU_WIDTH }}>
          <Grid container direction="column" spacing={3}>
            <Grid item key="close-menu">
              <IconButton onClick={props.onMenuClose}>
                <CloseIcon />
              </IconButton>
              <Divider />
            </Grid>
            <Grid
              item
              key="menu-items"
              container
              direction="column"
              spacing={3}
              style={{ marginLeft: 3 }}
            >
              <Grid item key="grouping-form" xs="auto" container>
                <Card variant="outlined">
                  <CardContent>
                    <Grid item container alignItems="center" spacing={6}>
                      <Grid item xs={6}>
                        <TextField
                          value={groupName}
                          error={
                            Object.keys(props.selected_objects).length <= 1
                          }
                          helperText={
                            Object.keys(props.selected_objects).length <= 1
                              ? "Select multiple objects"
                              : Object.keys(props.selected_objects).length +
                                " objects selected"
                          }
                          placeholder={
                            "Group Name " +
                            (groupingCount > 0 ? groupingCount + 1 : "")
                          }
                          onChange={(e) => {
                            setGroupName(e.target.value);
                          }}
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <Button
                          variant="contained"
                          disabled={
                            Object.keys(props.selected_objects).length <= 1 ||
                            !groupName
                          }
                          onClick={() => {
                            props.onGroupSubmit({
                              selected_objects: props.selected_objects,
                              group_name: groupName,
                            });
                            setGroupingCount((prev) => prev + 1);
                            setGroupName("");
                          }}
                        >
                          Group
                        </Button>
                      </Grid>
                      <Grid item xs={3}>
                        <Tooltip
                          title={groupingHelpText}
                          interactive
                          leaveDelay={500}
                        >
                          <HelpIcon />
                        </Tooltip>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              <Grid
                item
                key="options"
                container
                direction="column"
                alignItems="flex-start"
              >
                <Grid item key="toggle dyna pos" container alignItems="center">
                  <Grid item>
                    <FormControlLabel
                      control={
                        <Checkbox
                          size="small"
                          checked={props.dynamicPositioning}
                          onChange={props.toggleDynamicPositioning}
                        />
                      }
                      label="Dynamic Positioning"
                      labelPlacement="end"
                    />
                  </Grid>
                  <Grid item>
                    <Tooltip
                      title={dynamicPosHelpText}
                      placement="top"
                      interactive
                      leaveDelay={500}
                    >
                      <HelpIcon fontSize="small" />
                    </Tooltip>
                  </Grid>
                </Grid>
                <Grid item key="toggle show triples">
                  <FormControlLabel
                    control={
                      <Checkbox
                        size="small"
                        checked={props.showTriples}
                        onChange={props.toggleShowTriples}
                      />
                    }
                    label="Show triples"
                    labelPlacement="end"
                  />
                </Grid>
                <Grid item key="toggle square map">
                  <FormControlLabel
                    control={
                      <Checkbox
                        size="small"
                        checked={props.squareMap}
                        onChange={props.toggleSquareMap}
                      />
                    }
                    label="Square Map"
                    labelPlacement="end"
                  />
                </Grid>
              </Grid>
              <Grid item key="change-map-view" container alignItems="center">
                <Grid item>
                  {"View:    "}
                  <ButtonGroup variant="contained">
                    <Button
                      color={"ZX" === mapView ? "secondary" : "default"}
                      onClick={() => {
                        props.toggleMapView("ZX");
                        setMapView("ZX");
                      }}
                    >
                      ZX
                    </Button>
                    <Button
                      color={"XY" === mapView ? "secondary" : "default"}
                      onClick={() => {
                        props.toggleMapView("XY");
                        setMapView("XY");
                      }}
                    >
                      XY
                    </Button>
                    <Button
                      color={"YZ" === mapView ? "secondary" : "default"}
                      onClick={() => {
                        props.toggleMapView("YZ");
                        setMapView("YZ");
                      }}
                    >
                      YZ
                    </Button>
                  </ButtonGroup>
                </Grid>
                <Grid item>
                  <Button onClick={props.centerToBot}>CENTER TO BOT</Button>
                </Grid>
              </Grid>
              <Grid item key="color-picker" xs={9}>
                <Accordion styles={{ color: "#0000FF" }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Grid container alignItems="center" spacing={3}>
                      <Grid item>Node Details</Grid>
                      <Grid item>
                        <Tooltip
                          title={nodeDetailsHelpText}
                          placement="top"
                          interactive
                          leaveDelay={500}
                        >
                          <HelpIcon fontSize="small" />
                        </Tooltip>
                      </Grid>
                    </Grid>
                  </AccordionSummary>
                  {Object.entries(props.nodeTypeInfo).map(([type, info]) => (
                    <AccordionDetails key={type}>
                      <NodeColorPicker
                        type={type}
                        count={info.count}
                        color={info.color}
                        setNodeColoring={(color) => {
                          props.setNodeColoring(type, color);
                        }}
                      />
                    </AccordionDetails>
                  ))}
                </Accordion>
              </Grid>
            </Grid>
          </Grid>
        </div>
      </Drawer>
    </ThemeProvider>
  );
}
