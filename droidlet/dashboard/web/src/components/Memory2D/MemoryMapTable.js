import React from "react";
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

const MAX_TABLE_CELL_WIDTH = 100;

const StyledTableCell = withStyles((theme) => ({
  head: {
    backgroundColor: theme.palette.common.black,
    color: theme.palette.common.white,
  },
  body: {
    fontSize: 14,
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

const useStyles = makeStyles({
  table: {
    maxWidth: 300,
  },
  th: {
    maxWidth: MAX_TABLE_CELL_WIDTH,
    overflow: "hidden",
  },
});

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {rows, onTableDone} props
 *                            rows: pairs of memory attributes and value
 *                            onTableDone: event handler for after user is finished with table.
 */
export default function MemoryMapTable(props) {
  const classes = useStyles();

  return (
    <TableContainer component={Paper}>
      <Table className={classes.table} size="small" aria-label="a dense table">
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
        <TableBody>
          {props.rows.map((row) => (
            <StyledTableRow key={row.name}>
              <StyledTableCell
                className={classes.th}
                component="th"
                scope="row"
              >
                {shortenLongTableEntries(row.attribute)}
              </StyledTableCell>
              <StyledTableCell
                className={classes.th}
                component="th"
                scope="row"
              >
                {shortenLongTableEntries(row.value)}
              </StyledTableCell>
              <StyledTableCell
                className={classes.th}
                component="th"
                scope="row"
              >
                <IconButton disableRipple>
                  <DeleteIcon />
                </IconButton>
              </StyledTableCell>
            </StyledTableRow>
          ))}
          <StyledTableRow key={"onTableDone"}>
            <StyledTableCell
              className={classes.th}
              component="th"
              scope="row"
              colSpan={3}
              align="center"
            >
              <Button
                variant="contained"
                onClick={(e) => {
                  props.onTableDone(e);
                }}
              >
                Done
              </Button>
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
