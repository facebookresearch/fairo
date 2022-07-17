import React from "react";
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
 * @param {showMenu, onMenuClose, selected_objects, onGroupSubmit, dynamicPositioning, toggleDynamicPositioning, showTriples, toggleShowTriples} props
 *                            showMenu: bool for if menu should be open/close
 *                            onMenuClose: handler to close menu
 *                            selected_objects: dict of all objects selected and their data
 *                            onGroupSubmit: handler to submit grouping
 *                            dynamicPositioning: bool for if map tabular elements should dynamically
 *                                                position themselves based on window position
 *                            toggleDynamicPositioning: handler to toggle DP
 *                            showTriples: bool for if MemoryMapTable should show triples assoc with object
 *                            toggleShowTriples: handler to toggle showTriples
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
              <Switch
                size="small"
                checked={props.dynamicPositioning}
                onChange={props.toggleDynamicPositioning}
              />
            }
            label="Dynamic Positioning: "
            labelPlacement="end"
          />
          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={props.showTriples}
                onChange={props.toggleShowTriples}
              />
            }
            label="Show triples: "
            labelPlacement="end"
          />
        </div>
      </Drawer>
    </ThemeProvider>
  );
}
