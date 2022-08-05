/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D/MemoryMapTable.js

import React, { useEffect, useState } from "react";

import * as M2DC from "./Memory2DConstants";

import { withStyles } from "@material-ui/core/styles";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import Box from "@material-ui/core/Box";
import Button from "@material-ui/core/Button";
import IconButton from "@material-ui/core/IconButton";
import RefreshIcon from "@material-ui/icons/Refresh";
import RestoreIcon from "@material-ui/icons/Restore";
import CloseIcon from "@material-ui/icons/Close";
import Tooltip from "@material-ui/core/Tooltip";
import TextField from "@material-ui/core/TextField";

const MAX_TABLE_CELL_WIDTH = 100;
const MAX_TABLE_CONTAINER_HEIGHT = 220;
const MEMORY_MAP_TABLE_MAX_ENTRY_LENGTH = 16;

const StyledTableCell = withStyles((theme) => ({
  root: {
    fontSize: 14,
    fontFamily: M2DC.FONT,
    width: "auto !important",
    maxWidth: MAX_TABLE_CELL_WIDTH,
    overflow: "hidden",
  },
  head: {
    backgroundColor: theme.palette.common.black,
    color: theme.palette.common.white,
  },
  body: {
    color: theme.palette.common.black,
    borderBottomColor: theme.palette.common.black,
  },
}))(TableCell);

const StyledTableRow = withStyles((theme) => ({
  root: {
    "&:nth-of-type(odd)": {
      backgroundColor: theme.palette.action.hover,
    },
  },
}))(TableRow);

const StyledTextField = withStyles((theme) => ({
  root: {
    border: "1px solid #e2e2e1",
    overflow: "hidden",
    borderRadius: 4,
    backgroundColor: "#0a0a01",
    "& .Mui-error": {
      backgroundColor: "red",
    },
  },
}))(TextField);

function MapTextField(props) {
  const [value, setValue] = useState(props.value);
  const [errorCond, setErrorCond] = useState(false);

  return (
    <StyledTextField
      defaultValue={props.value}
      size="small"
      error={errorCond}
      onChange={(e) => {
        setValue(e.target.value);
        if (!e.target.value) {
          setErrorCond(true);
          props.updateDisableSubmit(true);
          return;
        } else {
          setErrorCond(false);
          props.updateDisableSubmit(false);
        }
        try {
          JSON.parse(e.target.value);
        } catch {
          setErrorCond(true);
          props.updateDisableSubmit(true);
        }
      }}
      onBlur={(e) => {
        if (value !== props.value) {
          props.updateStatus("changed");
        } else {
          props.updateStatus("same");
        }
        try {
          JSON.parse(value);
          props.updateValue(JSON.parse(value));
        } catch {
          props.updateStatus("error");
        }
      }}
      InputProps={{
        disableUnderline: true,
      }}
      inputProps={{
        style: {
          padding: 5,
          fontSize: 14,
          fontFamily: M2DC.FONT,
        },
      }}
    />
  );
}

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {data, onTableClose, onTableSubmit, onTableRestore, allTriples} props
 *                            data: dictionary of attribute: value pairs for object
 *                            onTableClose: event handler to close the table
 *                            onTableSubmit: event handler to send changed values to memory and close table
 *                            onTableRestore: event handler to restore table values to what they were without manual edits
 *                            allTriples: if show_triples toggled in menu, reference to all triples sent from agent memory
 */
export default function MemoryMapTable(props) {
  /*
  The editManager handles the state of the table in the form of a dictionary of the rows.
  Keys are row attributes.
  Value is object [
    value: <value>,
    valueType: <value_type>,
    status: "same" or "changed" or "error" [or "new" or "deleted"] [for future],
  ] 
  */
  const [memid, setMemid] = useState(null);
  const [editManager, setEditManager] = useState({});
  const [refresher, setRefresher] = useState(0);
  const [disableSubmit, setDisableSubmit] = useState(true);
  const [triples, setTriples] = useState([]);

  // update table values on data change/refresh
  useEffect(() => {
    if (props.data) {
      let em = {};
      Object.keys(props.data).forEach(
        (attr) =>
          (em[attr] = {
            value: props.data[attr],
            valueType: typeof props.data[attr],
            status: "same",
          })
      );
      setMemid(props.data["memid"]);
      setEditManager(em);
      setDisableSubmit(true);
    }
  }, [props.data, refresher]);

  // filter relevant triples to object
  useEffect(() => {
    if (props.allTriples) {
      let relevantTriples = [];
      props.allTriples.forEach((triple) => {
        if (triple[1] === memid) relevantTriples.push(triple);
      });
      setTriples(relevantTriples);
    }
  }, [props.allTriples, memid]);

  let immutableFields = ["memid", "eid", "node_type", "obj_id"];

  return (
    <TableContainer
      component={Paper}
      square
      style={{ maxHeight: MAX_TABLE_CONTAINER_HEIGHT }}
    >
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <StyledTableCell>Attribute</StyledTableCell>
            <StyledTableCell>
              <Box display="flex" justifyContent="space-between">
                Value
                <IconButton
                  onClick={props.onTableClose}
                  color="secondary"
                  size="small"
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Box>
            </StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody key={memid}>
          {Object.keys(
            Object.keys(editManager).reduce((toDisplay, attr) => {
              if (editManager[attr].status !== "deleted")
                toDisplay[attr] = editManager[attr];
              return toDisplay;
            }, {})
          ).map((attr) => (
            <StyledTableRow key={attr}>
              <StyledTableCell desc="attribute cell">
                {M2DC.shortenLongTableEntries(
                  attr,
                  MEMORY_MAP_TABLE_MAX_ENTRY_LENGTH,
                  "right-start"
                )}
              </StyledTableCell>
              <StyledTableCell desc="value cell">
                {immutableFields.includes(attr) ? (
                  M2DC.shortenLongTableEntries(
                    editManager[attr].value,
                    MEMORY_MAP_TABLE_MAX_ENTRY_LENGTH,
                    "right-start"
                  )
                ) : (
                  <MapTextField
                    attr={attr}
                    key={[editManager[attr].value, editManager[attr].status]}
                    value={JSON.stringify(editManager[attr].value)}
                    updateValue={(newValue) => {
                      setEditManager((prevEM) => ({
                        ...prevEM,
                        [attr]: {
                          ...prevEM[attr],
                          value: newValue,
                        },
                      }));
                    }}
                    updateStatus={(newStatus) => {
                      setEditManager((prevEM) => ({
                        ...prevEM,
                        [attr]: {
                          ...prevEM[attr],
                          status: newStatus,
                        },
                      }));
                    }}
                    updateDisableSubmit={(error) => {
                      error
                        ? setDisableSubmit((prev) => true)
                        : setDisableSubmit((prev) => false);
                    }}
                  />
                )}
              </StyledTableCell>
            </StyledTableRow>
          ))}
          <StyledTableRow>
            <StyledTableCell colSpan={2} align="center">
              <Box display="flex" justifyContent="space-around">
                <Tooltip
                  title="restores values to what they were before any manual edits"
                  placement="bottom"
                  interactive
                  leaveDelay={500}
                >
                  <IconButton
                    onClick={() => {
                      props.onTableRestore(props.data["memid"]);
                    }}
                    size="small"
                  >
                    <RestoreIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Button
                  variant="contained"
                  onClick={() => {
                    props.onTableSubmit(
                      Object.keys(editManager).reduce((toSend, attr) => {
                        // only send immutable, changed fields
                        if (
                          immutableFields.includes(attr) ||
                          editManager[attr].status === "changed"
                        )
                          toSend[attr] = editManager[attr];
                        return toSend;
                      }, {})
                    );
                  }}
                  disabled={disableSubmit}
                >
                  Submit
                </Button>
                <Tooltip
                  title="refreshes values to what they were when table was opened"
                  placement="bottom"
                  interactive
                  leaveDelay={500}
                >
                  <IconButton
                    onClick={() => {
                      setRefresher((count) => count + 1);
                    }}
                    size="small"
                  >
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </StyledTableCell>
          </StyledTableRow>
        </TableBody>
        {props.allTriples && triples.length > 0 && (
          <>
            <TableHead>
              <TableRow>
                <StyledTableCell>Predicate</StyledTableCell>
                <StyledTableCell>
                  <Box display="flex" justifyContent="space-between">
                    Value
                    <IconButton
                      onClick={props.onTableClose}
                      color="secondary"
                      size="small"
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {triples.map((triple) => (
                <StyledTableRow key={triple[0]}>
                  <StyledTableCell desc="predicate">
                    {M2DC.shortenLongTableEntries(
                      triple[4],
                      MEMORY_MAP_TABLE_MAX_ENTRY_LENGTH,
                      "right-start"
                    )}
                  </StyledTableCell>
                  <StyledTableCell desc="value">
                    {M2DC.shortenLongTableEntries(
                      triple[6],
                      MEMORY_MAP_TABLE_MAX_ENTRY_LENGTH,
                      "right-start"
                    )}
                  </StyledTableCell>
                </StyledTableRow>
              ))}
            </TableBody>
          </>
        )}
      </Table>
    </TableContainer>
  );
}

export function positionMemoryMapTable(
  h,
  w,
  tc,
  dc,
  makeDynamic = false,
  data = null
) {
  // this takes all these parameters so table will properly update position on change
  let ret = { position: "absolute" };
  let final_coords = [tc[0] + dc[0], tc[1] + dc[1]];
  let final_pos = ["left", "top"];
  if (makeDynamic) {
    let table_dims = [
      200,
      Math.min(
        MAX_TABLE_CONTAINER_HEIGHT - 10,
        42 * (Object.keys(data).length - 3) + 32 * 4 + 51
      ),
    ];
    if (final_coords[1] > Math.min(h, w) - table_dims[1]) {
      final_coords[1] = Math.min(h, w) - final_coords[1];
      final_pos[1] = "bottom";
    }
  }
  ret[final_pos[0]] = final_coords[0];
  ret[final_pos[1]] = final_coords[1];
  return ret;
}
