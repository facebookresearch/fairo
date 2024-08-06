/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D/ClusteredObjsPopup.js

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
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import LensIcon from "@material-ui/icons/Lens"; // Circle

const MAX_TABLE_CELL_WIDTH = 100;
const MAX_TABLE_CONTAINER_HEIGHT = 220;
const CLUSTERED_OBJS_POPUP_MAX_ENTRY_LENGTH = 12;

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
    borderBottomColor: "transparent",
  },
}))(TableCell);

const StyledTableRow = withStyles((theme) => ({
  root: {
    "&:nth-of-type(odd)": {
      backgroundColor: theme.palette.action.hover,
    },
    "&:hover": {
      backgroundColor: "orange",
    },
  },
}))(TableRow);

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {data, map_pos, onPopupClose, handleObjClick, table_visible, selection_mode, selected_objects} props
 *                            data: array of poolData objects
 *                            map_pos: where popup should be positioned on map
 *                            onPopupClose: handler for after user is finished with popup
 *                            handleObjClick: handler for clicking on a row (representing an object)
 *                            table_visible: used for hack to know when object is focused
 *                            selection_mode: prop to let component know when user has selectionKey pressed
 *                            selected_objects: dict of objects that were grouped by user
 */
export default function ClusteredObjsPopup(props) {
  const [focusedObj, setFocusedObj] = useState(null);

  useEffect(() => {
    if (!props.table_visible) setFocusedObj(null);
  }, [props.table_visible]);

  return (
    <TableContainer
      component={Paper}
      square
      style={{ maxHeight: MAX_TABLE_CONTAINER_HEIGHT }}
    >
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <StyledTableCell>Memid</StyledTableCell>
            <StyledTableCell>
              <Box display="flex" justifyContent="space-between">
                Pos
                <IconButton
                  onClick={(e) => {
                    props.onPopupClose(e);
                  }}
                  color="secondary"
                  size="small"
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Box>
            </StyledTableCell>
          </TableRow>
        </TableHead>

        <TableBody key={props.map_pos}>
          {props.data.map((poolData) => (
            <StyledTableRow
              key={poolData.data.memid}
              onClick={() => {
                props.handleObjClick(
                  poolData.type,
                  [props.map_pos[0], props.map_pos[1]],
                  poolData.data
                );
                if (!props.selection_mode) setFocusedObj(poolData.data.memid);
              }}
              style={
                poolData.data.memid === focusedObj
                  ? { backgroundColor: "orange" }
                  : poolData.data.memid in props.selected_objects
                  ? { backgroundColor: "green" }
                  : {}
              }
            >
              <StyledTableCell>
                <Box
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  {M2DC.shortenLongTableEntries(
                    poolData.data.memid,
                    CLUSTERED_OBJS_POPUP_MAX_ENTRY_LENGTH
                  )}
                  <IconButton size="small" disableRipple>
                    <LensIcon
                      style={{ color: poolData.color }}
                      fontSize="small"
                    />
                  </IconButton>
                </Box>
              </StyledTableCell>
              <StyledTableCell>
                {" "}
                {JSON.stringify(poolData.data.pos)}{" "}
              </StyledTableCell>
            </StyledTableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export function positionClusteredObjsPopup(
  h,
  w,
  tc,
  dc,
  makeDynamic = false,
  data = null
) {
  // this takes all these parameters so table will properly update position on change
  let ret = { position: "absolute" };
  let final_coords = [w - (tc[0] + dc[0]), tc[1] + dc[1]];
  let final_pos = ["right", "top"];
  if (makeDynamic) {
    let table_dims = [
      200,
      Math.min(MAX_TABLE_CONTAINER_HEIGHT - 10, 32 * data.length + 28),
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
