import React, { useEffect, useState } from "react";
import Button from "@material-ui/core/Button";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import Drawer from "@material-ui/core/Drawer";
import Divider from "@material-ui/core/Divider";
import TextField from "@material-ui/core/TextField";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Switch from "@material-ui/core/Switch";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

const menuTheme = createMuiTheme({
  palette: {
    type: "dark",
  },
});

/**
 * Creates simple table of memory values for an object on the map.
 *
 * @param {data, onTableClose, onTableSubmit} props
 *                            data: dictionary of attribute: value pairs for object
 *                            onTableClose: event handler to close the table
 *                            onTableSubmit: event handler to send changed values to memory and close table
 */
export default function Memory2DMenu(props) {
  const [groupName, setGroupName] = React.useState("");
  const [groupingCount, setGroupingCount] = React.useState(0);

  return (
    <ThemeProvider theme={menuTheme}>
      <Drawer anchor="right" open={props.showMenu} onClose={props.onMenuClose}>
        <div style={{ width: 450 }}>
          <IconButton onClick={props.onMenuClose}>
            <CloseIcon />
          </IconButton>
          <Divider />
          <TextField
            value={groupName}
            disabled={Object.keys(props.selected_objects).length <= 1}
            placeholder={
              "groupName" + (groupingCount > 0 ? groupingCount + 1 : "")
            }
            onChange={(e) => {
              setGroupName(e.target.value);
            }}
          />
          <Button
            variant="contained"
            disabled={
              Object.keys(props.selected_objects).length <= 1 || !groupName
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
          <FormControlLabel
            control={
              <Switch size="small" onChange={props.toggleDynamicPositioning} />
            }
            label="Dynamic Positioning: "
            labelPlacement="end"
          />
        </div>
      </Drawer>
    </ThemeProvider>
  );
}
