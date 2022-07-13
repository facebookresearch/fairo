import React, { useEffect, useState } from "react";
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
import ClearIcon from "@material-ui/icons/Clear";
import Tooltip from "@material-ui/core/Tooltip";

import TextField from "@material-ui/core/TextField";

const MAX_TABLE_CELL_WIDTH = 100;

const StyledTableCell = withStyles((theme) => ({
  root: {
    fontSize: 14,
    fontFamily: "Segoe UI",
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
          fontFamily: "Segoe UI",
        },
      }}
    />
  );
}

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {data, onTableClose, onTableSubmit} props
 *                            data: dictionary of attribute: value pairs for object
 *                            onTableClose: event handler to close the table
 *                            onTableSubmit: event handler to send changed values to memory and close table
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
  const [editManager, setEditManager] = useState({});
  const [refresher, setRefresher] = useState(0);
  const [disableSubmit, setDisableSubmit] = useState(true);

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
      setEditManager(em);
      setDisableSubmit(true);
    }
  }, [props.data, refresher]);

  let immutableFields = ["memid", "eid", "node_type", "obj_id"];

  // console.log(editManager);
  return (
    <TableContainer component={Paper} square>
      <Table size="small">
        <TableHead>
          <TableRow>
            <StyledTableCell>Attribute</StyledTableCell>
            <StyledTableCell>
              <Box display="flex" justifyContent="space-between">
                Value
                <IconButton
                  onClick={() => {
                    props.onTableClose();
                  }}
                  color="secondary"
                  size="small"
                >
                  <ClearIcon fontSize="small" />
                </IconButton>
              </Box>
            </StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody key={props.data["memid"]}>
          {Object.keys(
            Object.keys(editManager).reduce((toDisplay, attr) => {
              if (editManager[attr].status !== "deleted")
                toDisplay[attr] = editManager[attr];
              return toDisplay;
            }, {})
          ).map((attr) => (
            <StyledTableRow key={attr}>
              <StyledTableCell desc="attribute cell">
                {shortenLongTableEntries(attr)}
              </StyledTableCell>
              <StyledTableCell desc="value cell">
                {immutableFields.includes(attr) ? (
                  shortenLongTableEntries(editManager[attr].value)
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
                <Button
                  variant="contained"
                  onClick={(e) => {
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
                <IconButton
                  onClick={(e) => {
                    setRefresher((count) => count + 1);
                  }}
                  size="small"
                >
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Box>
            </StyledTableCell>
          </StyledTableRow>
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export function positionMemoryMapTable(h, w, tc, dc, data) {
  // this takes all these parameters so table will properly update position on change
  let ret = { position: "absolute" };
  let final_coords = [tc[0] + dc[0], Math.min(h, w) - (tc[1] + dc[1])];
  let final_pos = ["left", "bottom"];
  let table_dims = [200, 42 * (Object.keys(data).length - 3) + 32 * 4 + 51];
  if (final_coords[1] > Math.min(h, w) - table_dims[1]) {
    final_coords[1] = Math.min(h, w) - final_coords[1];
    final_pos[1] = "top";
  }
  ret[final_pos[0]] = final_coords[0];
  ret[final_pos[1]] = final_coords[1];
  return ret;
}

function shortenLongTableEntries(e) {
  if (e.length > 16) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        maxHeight={20}
      >
        <Tooltip title={e} placement="right-start" interactive leaveDelay={500}>
          <p>{e.substring(0, 6) + "..." + e.substring(e.length - 6)}</p>
        </Tooltip>
      </Box>
    );
  }
  return e;
}
