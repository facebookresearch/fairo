/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * Labeling.js handles the table where users can correct the categories and spans
 */
import React, { useState } from "react";
import { makeStyles } from "@material-ui/core/styles";

import Input from "@material-ui/core/Input";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import Select from "@material-ui/core/Select";
import Button from "@material-ui/core/Button";

//table
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";

const attributes = [
  "",
  "has_colour",
  "has_size",
  "has_name",
  "has_orientation",
  "has_thickness",
  "has_height",
  "has_length",
  "has_radius",
  "has_slope",
  "has_width",
  "has_base",
  "has_distance",
  "has_block_type",
];
const dropdown_choices = {
  dialogue_type: ["HUMAN_GIVE_COMMAND", "get_memory", "put_memory"],
  action_type: [
    "BUILD",
    "COPY",
    "NOOP",
    "SPAWN",
    "RESUME",
    "FILL",
    "DESTROY",
    "MOVE",
    "UNDO",
    "STOP",
    "DIG",
    "FREEBUILD",
    "DANCE",
  ],
  relative_direction: [
    "LEFT",
    "RIGHT",
    "DOWN",
    "FRONT",
    "BACK",
    "AWAY",
    "INSIDE",
    "NEAR",
    "OUTSIDE",
  ],
  location_type: [
    "COORDINATES",
    "REFERENCE_OBJECT",
    "AGENT_POS",
    "SPEAKER_POS",
    "SPEAKER_LOOK",
  ],
};

const useStyles = makeStyles((theme) => ({
  category: {
    "font-weight": "bold",
  },
}));

function Labeling({ actiondict, message, goToEnd }) {
  const [state, setState] = React.useState({
    addict: { actiondict },
  });

  const classes = useStyles();

  function renderMenuItems(arr) {
    //drop down selects, show the other options
    return arr.map((value) =>
      React.cloneElement(<MenuItem value={value}>{value}</MenuItem>)
    );
  }

  function handleChangeKey(event, prev) {
    //updates the state action dictionary when a key is changed (for the has_ labels)
    function updateKey(dict, prev, newkey) {
      if (prev.length > 1) {
        var next = prev.shift();
        dict[next.toString()] = updateKey(dict[next], prev, newkey);
      } else {
        var val = dict[prev[0]];
        delete dict[prev[0]];
        dict[newkey] = val;
      }
      return dict;
    }
    var newstate = state.addict;
    newstate = updateKey(newstate, prev, event.target.value);
    setState((oldValues) => ({ addict: newstate }));
  }

  function handleChangeValue(event, prev) {
    //updates the state action dictionary when a value is changed
    function updateValue(dict, prev, newval) {
      if (prev.length > 1) {
        var next = prev.shift();
        dict[next.toString()] = updateValue(dict[next], prev, newval);
      } else {
        dict[prev[0]] = newval;
      }
      return dict;
    }
    var newstate = state.addict;
    newstate = updateValue(newstate, prev, event.target.value);
    setState((oldValues) => ({ addict: newstate }));
  }

  function handleAddKey(event, prev) {
    //adds a new key to the action dictionary
    function addKey(dict, prev, newkey) {
      if (prev.length > 1) {
        var next = prev.shift();
        dict[next.toString()] = addKey(dict[next], prev, newkey);
      } else {
        dict[newkey] = "";
      }
      return dict;
    }

    var newstate = state.addict;
    newstate = addKey(newstate, prev, event.target.value);
    setState((oldValues) => ({ addict: newstate }));
  }

  function getVal(dict, prev) {
    //get the value of a chain of keys in an array (prev) from the nested dict
    if (prev.length > 1) {
      var next = prev.shift();
      return getVal(dict[next], prev);
    } else {
      return dict[prev[0]].toString();
    }
  }

  function doneClicked() {
    //go to last view of question flow and pass back the new actiondictionary
    goToEnd(state.addict.actiondict);
  }

  function renderRows(dict, col, prevkeys) {
    //display the rows of the labeling table
    var rows = [];
    Object.keys(dict).forEach(function (key) {
      if (typeof dict[key] == "object") {
        //recurse
        rows = rows.concat(
          renderRows(dict[key], col + 1, prevkeys.concat(key))
        );
      } else {
        var r = [];
        for (var i = 0; i < col - 1; i++) {
          if (i === col - 2 && prevkeys.length > 0) {
            r.push(
              <TableCell
                align="center"
                className={classes.category}
                scope="row"
              >
                {prevkeys[prevkeys.length - 1]}
              </TableCell>
            );
          } else {
            r.push(<TableCell align="center" scope="row"></TableCell>);
          }
        }
        //if schematic
        if (key.includes("has_")) {
          r.push(
            <TableCell align="center" scope="row">
              <FormControl>
                <Select
                  value={key}
                  onChange={(event) =>
                    handleChangeKey(event, prevkeys.slice().concat(key))
                  }
                  input={<Input />}
                >
                  {renderMenuItems(attributes)}
                </Select>
              </FormControl>
            </TableCell>
          );
        } else {
          r.push(
            <TableCell align="center" component="th" scope="row">
              {key}
            </TableCell>
          );
        }

        //check if need dropdown or text edit
        if (key in dropdown_choices) {
          r.push(
            <TableCell align="center">
              <FormControl>
                <Select
                  value={getVal(state.addict, prevkeys.slice().concat(key))}
                  onChange={(event) =>
                    handleChangeValue(event, prevkeys.slice().concat(key))
                  }
                  input={<Input />}
                  inputProps={{
                    prevkeys: prevkeys.slice().concat(key),
                  }}
                >
                  {renderMenuItems(dropdown_choices[key])}
                </Select>
              </FormControl>
            </TableCell>
          );
        } else {
          r.push(
            <TableCell align="center">
              <Input
                defaultValue={getVal(
                  state.addict,
                  prevkeys.slice().concat(key)
                )}
                onChange={(event) =>
                  handleChangeValue(event, prevkeys.slice().concat(key))
                }
                inputProps={{
                  "aria-label": "Description",
                }}
              />
            </TableCell>
          );
        }
        rows = rows.concat(<TableRow>{r}</TableRow>);
        if (dict[key].includes(" ")) {
          r = [];
          for (var j = 0; j < col; j++) {
            r.push(
              <TableCell align="center" component="th" scope="row"></TableCell>
            );
          }
          r.push(
            <TableCell>
              Do any of the words describe the other? <br></br>
              <FormControl>
                <Select
                  onChange={(event) =>
                    handleAddKey(event, prevkeys.slice().concat(key))
                  }
                  input={<Input />}
                >
                  {renderMenuItems(attributes)}
                </Select>
              </FormControl>
            </TableCell>
          );
          rows = rows.concat(<TableRow>{r}</TableRow>);
        }
      }
    });
    return rows;
  }

  return (
    <div>
      <Paper>
        <Table>
          <TableBody>{renderRows(state.addict, 0, [])}</TableBody>
        </Table>
      </Paper>
      <Button variant="contained" color="primary" onClick={doneClicked}>
        Done
      </Button>
    </div>
  );
}

export default Labeling;
