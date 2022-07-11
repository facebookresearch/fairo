import React, { useEffect, useState } from "react";
import { withStyles, makeStyles } from "@material-ui/core/styles";
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
    borderBottomColor: "transparent",
  },
}))(TableCell);

const StyledTableRow = withStyles((theme) => ({
  root: {
    "&:nth-of-type(odd)": {
      backgroundColor: theme.palette.action.hover,
    },
  },
}))(TableRow);

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {rows, onTableDone} props
 *                            rows: pairs of memory attributes and value
 *                            onTableDone: event handler for after user is finished with table.
 */
export default function MemoryPopup(props) {
  /*
  The editManager handles the state of the table in the form of a dictionary of the rows.
  Keys are row attributes.
  Value is object [
    value: <value>,
    valueType: <value_type>,
    status: "same" or "changed" or "error" [or "new" or "deleted"] [for future],
  ] 
  */
  const [rows, setRows] = useState({});

  useEffect(() => {
    if (props.data) {
    }
  }, [props.data]);

  // console.log(editManager);
  return (
    <TableContainer component={Paper} square>
      <Table size="small">
        <TableHead>
          <TableRow>
            <StyledTableCell>Memid</StyledTableCell>
            <StyledTableCell>
              <Box display="flex" justifyContent="space-between">
                Position
                <IconButton
                  onClick={(e) => {
                    props.onPopupClose(e);
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

        <TableBody
          key={
            // props.map_pos
            "temp"
          }
        >
          {props.data.map((poolData) => (
            <StyledTableRow>
              <StyledTableCell>
                {shortenLongTableEntries(poolData.data.memid)}
              </StyledTableCell>
              <StyledTableCell>
                {JSON.stringify(poolData.data.pos)}
              </StyledTableCell>
            </StyledTableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
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
        <Tooltip title={e} placement="left-start" interactive leaveDelay={500}>
          <p>{e.substring(0, 6) + "..." + e.substring(e.length - 6)}</p>
        </Tooltip>
      </Box>
    );
  }
  return e;
}
