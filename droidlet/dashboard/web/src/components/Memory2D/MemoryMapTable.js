import React, { useEffect, useState } from "react";
import { withStyles, makeStyles } from "@material-ui/core/styles";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";

import Button from "@material-ui/core/Button";
import IconButton from "@material-ui/core/IconButton";
import DeleteIcon from "@material-ui/icons/Delete";
import ClearIcon from "@material-ui/icons/Clear";
import AddIcon from "@material-ui/icons/Add";

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
  },
}))(TextField);

const useStyles = makeStyles((theme) => ({
  table: {
    maxWidth: 400,
  },
  cell: {
    root: {
      fontSize: 14,
      fontFamily: "Segoe UI",
      maxWidth: MAX_TABLE_CELL_WIDTH,
      overflow: "hidden",
    },
    head: {
      backgroundColor: theme.palette.common.white,
      color: theme.palette.common.white,
    },
    body: {
      color: theme.palette.common.black,
    },
  },
}));

function MyTextField(props) {
  const [value, setValue] = useState(props.value);

  return (
    <StyledTextField
      defaultValue={props.value}
      margin="normal"
      error={!value}
      onChange={(e) => {
        setValue(e.target.value);
      }}
      onFocus={(e) => {
        console.log("focused");
      }}
      onBlur={(e) => {
        if (value !== props.value) {
          console.log("changed");
        } else {
          console.log("same");
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
 * @param {rows, onTableDone} props
 *                            rows: pairs of memory attributes and value
 *                            onTableDone: event handler for after user is finished with table.
 */
export default function MemoryMapTable(props) {
  const classes = useStyles();

  /*
  The editManager handles the state of the table in the form of a dictionary of the rows.
  Keys are row attributes.
  Value is array [
    <value>,
    "orig" or "new",
    "keep" or "delete"
  ] 
  */
  const [editManager, setEditManager] = useState({});

  // clicking on new object updates component state
  useEffect(() => {
    if (props.data) {
      let em = {};
      Object.keys(props.data).forEach(
        (attr) => (em[attr] = [props.data[attr], "orig", "keep"])
      );
      setEditManager(em);
    }
  }, [props.data]);

  let immutableFields = ["memid"];

  // console.log(editManager);
  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <StyledTableCell>Attribute</StyledTableCell>
            <StyledTableCell>Value</StyledTableCell>
            <StyledTableCell>
              <IconButton
                onClick={(e) => {
                  props.onTableDone(e);
                }}
                color="secondary"
              >
                <ClearIcon />
              </IconButton>
            </StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody key={props.data["memid"]}>
          {Object.keys(
            Object.keys(editManager).reduce((toDisplay, attr) => {
              // mutable fields
              if (editManager[attr][2] === "keep")
                toDisplay[attr] = editManager[attr];
              return toDisplay;
            }, {})
          ).map((attr) => (
            <StyledTableRow key={attr}>
              <StyledTableCell>
                {" "}
                {shortenLongTableEntries(attr)}{" "}
              </StyledTableCell>
              <StyledTableCell>
                {immutableFields.includes(attr) ? (
                  shortenLongTableEntries(editManager[attr][0])
                ) : (
                  <MyTextField attr={attr} value={editManager[attr][0]} />
                )}
              </StyledTableCell>
              <StyledTableCell>
                {!immutableFields.includes(attr) && (
                  <IconButton
                    onClick={(e) => {
                      setEditManager((prevEM) => ({
                        ...prevEM,
                        [attr]: prevEM[attr].slice(0, 2).concat(["delete"]),
                      }));
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                )}
              </StyledTableCell>
            </StyledTableRow>
          ))}
          <StyledTableRow key={"onTableDone"}>
            <StyledTableCell colSpan={2} align="center">
              <Button
                variant="contained"
                onClick={(e) => {
                  props.onTableDone(e);
                }}
              >
                Submit
              </Button>
            </StyledTableCell>
            <StyledTableCell>
              <IconButton>
                <AddIcon />
              </IconButton>
            </StyledTableCell>
          </StyledTableRow>
        </TableBody>
      </Table>
    </TableContainer>
  );
}

function shortenLongTableEntries(e) {
  if (e.length > 16) {
    return e.substring(0, 6) + "..." + e.substring(e.length - 6);
  }
  return e;
}
