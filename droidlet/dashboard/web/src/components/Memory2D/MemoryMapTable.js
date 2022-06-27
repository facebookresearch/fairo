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
import Edit from "@material-ui/icons/Edit";
import DoneIcon from "@material-ui/icons/Done";
import InputAdornment from "@material-ui/core/InputAdornment";

import TextField from "@material-ui/core/TextField";
import { DialogContent } from "@material-ui/core";

const MAX_TABLE_CELL_WIDTH = 100;

const StyledTableCell = withStyles((theme) => ({
  head: {
    backgroundColor: theme.palette.common.black,
    color: theme.palette.common.white,
    fontFamily: "Segoe UI",
  },
  body: {
    fontSize: 14,
    color: theme.palette.common.black,
    fontFamily: "Segoe UI",
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
  etf: {
    textField: {
      color: "black",
      borderBottom: 0,
      "&:before": {
        borderBottom: 0,
      },
    },
    disabled: {
      color: "black",
      borderBottom: 0,
      "&:before": {
        borderBottom: 0,
      },
    },
  },
});

class EditableTextField extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      value: props.value,
      editMode: false,
      mouseEnter: false,
    };
    this.state = this.initialState;
  }

  handleChange = (e) => {
    this.setState({ value: e.target.value });
  };

  handleMouseEnter = (event) => {
    if (!this.state.mouseEnter) {
      this.setState({ mouseEnter: true });
    }
  };

  handleMouseLeave = (event) => {
    if (this.state.mouseEnter) {
      this.setState({ mouseEnter: false });
    }
  };

  handleClick = () => {
    if (!this.state.editMode || this.state.value) {
      // only allow exit of editMode when there is a value
      this.setState({ editMode: !this.state.editMode });
      //this.props.updateEditManager(this.props.attribute, willEdit);
    }
  };

  render() {
    const { attribute, value } = this.props;

    return (
      <TextField
        name="cell"
        defaultValue={value}
        margin="normal"
        error={!this.state.value}
        onChange={this.handleChange}
        disabled={!this.state.editMode}
        onMouseEnter={this.handleMouseEnter}
        onMouseLeave={this.handleMouseLeave}
        InputProps={{
          endAdornment: this.state.mouseEnter ? (
            <InputAdornment position="end">
              <IconButton onClick={this.handleClick}>
                {!this.state.editMode ? <Edit /> : <DoneIcon />}
              </IconButton>
            </InputAdornment>
          ) : (
            ""
          ),
        }}
      />
    );
  }
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

  const [data, setData] = useState(props.data);
  const [editManager, setEditManager] = useState({});
  const [changes, setChanges] = useState({});

  // useEffect(() => {
  //   if(props.data) {
  //     setChanges(Object.entries(props.data));
  //   }
  // }, [props.data])

  useEffect(() => {
    let em = {};
    Object.keys(props.data).forEach((attribute) => (em[attribute] = false));
    setEditManager(em);
  }, []);

  // function updateEditManager(attribute, willEdit) {
  //   const { [attribute]: omitted, ...rest } = editManager;
  //   if (Object.values(rest).every((val) => val === false)) {
  //     setEditManager(prevEditManager => ({
  //       ...prevEditManager,
  //       [attribute]: willEdit,
  //     }));
  //   }
  //   console.log(editManager);
  // };

  /*
  const testKey = document.getElementById('test');
  testKey.addEventListener('focusin', (event) => {
    event.target.style.background = 'pink';
  });
  
  testKey.addEventListener('focusout', (event) => {
    event.target.style.background = '';
  });  
  */

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
          <StyledTableRow key="test_input_row">
            <StyledTableCell key="something">Let's go</StyledTableCell>
            <StyledTableCell key="test_input">
              <EditableTextField attribute="Let's go" value="hahahahaha" s />
            </StyledTableCell>
          </StyledTableRow>
          {Object.keys(props.data).map((attribute) => (
            <StyledTableRow key={attribute}>
              <StyledTableCell
                className={classes.th}
                component="th"
                scope="row"
              >
                {shortenLongTableEntries(attribute)}
              </StyledTableCell>
              <StyledTableCell
                className={classes.th}
                component="th"
                scope="row"
              >
                {shortenLongTableEntries(props.data[attribute].toString())}
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
