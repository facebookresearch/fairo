import React from "react";
import Button from "@material-ui/core/Button";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import HelpIcon from "@material-ui/icons/Help";
import Tooltip from "@material-ui/core/Tooltip";
import Drawer from "@material-ui/core/Drawer";
import Divider from "@material-ui/core/Divider";
import TextField from "@material-ui/core/TextField";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Checkbox from "@material-ui/core/Checkbox";
import Grid from "@material-ui/core/Grid";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

const menuTheme = createMuiTheme({
  palette: {
    type: "dark",
  },
});

const groupingHelpText =
  "Group selected objects and store as 'is_a' predicate triple in memory. To select objects, use Cmd+Click or right click to start describing a selection window";

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
          <Grid container direction="column" spacing={3}>
            <Grid item>
              <IconButton onClick={props.onMenuClose}>
                <CloseIcon />
              </IconButton>
              <Divider />
            </Grid>
            <Grid item xs="auto">
              <Card variant="outlined">
                <CardContent>
                  <Grid
                    container
                    justifyContent="space-between"
                    alignItems="center"
                    spacing={2}
                  >
                    <Grid item xs={6}>
                      <TextField
                        value={groupName}
                        error={Object.keys(props.selected_objects).length <= 1}
                        helperText={
                          Object.keys(props.selected_objects).length <= 1
                            ? "Select multiple objects"
                            : Object.keys(props.selected_objects).length +
                              " objects selected"
                        }
                        placeholder={
                          "Group Name " +
                          (groupingCount > 0 ? groupingCount + 1 : "")
                        }
                        onChange={(e) => {
                          setGroupName(e.target.value);
                        }}
                      />
                    </Grid>
                    <Grid item xs={3}>
                      <Button
                        variant="contained"
                        disabled={
                          Object.keys(props.selected_objects).length <= 1 ||
                          !groupName
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
                    </Grid>
                    <Grid item xs={3}>
                      <Tooltip
                        title={groupingHelpText}
                        interactive
                        leaveDelay={500}
                      >
                        <HelpIcon />
                      </Tooltip>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
            <Grid item container direction="column" alignItems="flex-start">
              <Grid item>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={props.dynamicPositioning}
                      onChange={props.toggleDynamicPositioning}
                    />
                  }
                  label="Dynamic Positioning"
                  labelPlacement="end"
                />
              </Grid>
              <Grid item>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={props.showTriples}
                      onChange={props.toggleShowTriples}
                    />
                  }
                  label="Show triples"
                  labelPlacement="end"
                />
              </Grid>
            </Grid>
          </Grid>
        </div>
      </Drawer>
    </ThemeProvider>
  );
}
