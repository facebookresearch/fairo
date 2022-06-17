import React from "react";
import { withStyles, makeStyles } from "@material-ui/core/styles";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import IconButton from "@material-ui/core/IconButton";
import DeleteIcon from "@material-ui/icons/Delete";

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
    maxWidth: 200,
  },
  th: {
    maxWidth: 100,
    maxHeight: 55,
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
                {row.attribute}
              </StyledTableCell>
              <StyledTableCell
                className={classes.th}
                component="th"
                scope="row"
              >
                {row.value}
              </StyledTableCell>
            </StyledTableRow>
          ))}
          <StyledTableRow key={"onTableDone"}>
            <StyledTableCell
              className={classes.th}
              component="th"
              scope="row"
              colSpan={2}
              align="center"
            >
              <IconButton
                onClick={(e) => {
                  props.onTableDone(e);
                }}
              >
                <DeleteIcon />
              </IconButton>
            </StyledTableCell>
          </StyledTableRow>
        </TableBody>
      </Table>
    </TableContainer>
  );
}
