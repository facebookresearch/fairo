import React from "react";
import PropTypes from "prop-types";
import clsx from "clsx";
import { withStyles } from "@material-ui/core/styles";
import TableCell from "@material-ui/core/TableCell";
import Paper from "@material-ui/core/Paper";
import { AutoSizer, Column, Table } from "react-virtualized";

const styles = (theme) => ({
  flexContainer: {
    display: "flex",
    alignItems: "center",
    boxSizing: "border-box",
  },
  table: {
    // temporary right-to-left patch, waiting for
    // https://github.com/bvaughn/react-virtualized/issues/454
    "& .ReactVirtualized__Table__headerRow": {
      flip: false,
      paddingRight: theme.direction === "rtl" ? "0 !important" : undefined,
    },
  },
  tableRow: {
    cursor: "pointer",
  },
  tableRowHover: {
    "&:hover": {
      backgroundColor: theme.palette.grey[200],
    },
  },
  tableCell: {
    flex: 1,
  },
  noClick: {
    cursor: "initial",
  },
});

class MuiVirtualizedTable extends React.PureComponent {
  static defaultProps = {
    headerHeight: 48,
    rowHeight: 48,
  };

  getRowClassName = ({ index }) => {
    const { classes, onRowClick } = this.props;

    return clsx(classes.tableRow, classes.flexContainer, {
      [classes.tableRowHover]: index !== -1 && onRowClick != null,
    });
  };

  cellRenderer = ({ cellData, columnIndex }) => {
    const { columns, classes, rowHeight, onRowClick } = this.props;
    return (
      <TableCell
        component="div"
        className={clsx(classes.tableCell, classes.flexContainer, {
          [classes.noClick]: onRowClick == null,
        })}
        variant="body"
        style={{ height: rowHeight }}
        align={
          (columnIndex != null && columns[columnIndex].numeric) || false
            ? "right"
            : "left"
        }
      >
        {cellData}
      </TableCell>
    );
  };

  headerRenderer = ({ label, columnIndex }) => {
    const { headerHeight, columns, classes } = this.props;

    return (
      <TableCell
        component="div"
        className={clsx(
          classes.tableCell,
          classes.flexContainer,
          classes.noClick
        )}
        variant="head"
        style={{ height: headerHeight }}
        align={columns[columnIndex].numeric || false ? "right" : "left"}
      >
        <span>{label}</span>
      </TableCell>
    );
  };

  render() {
    const {
      height,
      width,
      classes,
      columns,
      rowHeight,
      headerHeight,
      ...tableProps
    } = this.props;

    console.log("MuiVirtualizedTable", height, width);
    return (
      // <AutoSizer style={{ height: { height }, width: { width } }}>
      //   {({ height, width }) => (
      //     <>
      //       <div>
      //         {" "}
      //         {height}, {width}{" "}
      //       </div>
      <Table
        height={height}
        width={width}
        rowHeight={rowHeight}
        gridStyle={{
          direction: "inherit",
        }}
        size="small"
        headerHeight={headerHeight}
        className={classes.table}
        {...tableProps}
        rowClassName={this.getRowClassName}
      >
        {columns.map(({ dataKey, ...other }, index) => {
          return (
            <Column
              key={dataKey}
              headerRenderer={(headerProps) =>
                this.headerRenderer({
                  ...headerProps,
                  columnIndex: index,
                })
              }
              className={classes.flexContainer}
              cellRenderer={this.cellRenderer}
              dataKey={dataKey}
              {...other}
            />
          );
        })}
      </Table>
      //     </>
      //   )}
      // </AutoSizer>
    );
  }
}

MuiVirtualizedTable.propTypes = {
  classes: PropTypes.object.isRequired,
  columns: PropTypes.arrayOf(
    PropTypes.shape({
      dataKey: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      numeric: PropTypes.bool,
      width: PropTypes.number.isRequired,
    })
  ).isRequired,
  headerHeight: PropTypes.number,
  onRowClick: PropTypes.func,
  rowHeight: PropTypes.number,
};

const VirtualizedTable = withStyles(styles)(MuiVirtualizedTable);

function toReferencedObject(arr) {
  return { uuid: arr[0], id: arr[1], type: arr[9], name: arr[7] };
}

export default function ReactVirtualizedTable({
  height,
  width,
  memoryManager,
  onShowMemeoryDetail,
}) {
  const [getCount, getMemoryForIndex] = memoryManager;

  return (
    // TODO setting the heingt to 100% is not working. Will need to figure this out.
    <Paper style={{ height: { height }, width: { width } }}>
      <VirtualizedTable
        height={height}
        width={width}
        rowCount={getCount()}
        rowGetter={({ index }) => toReferencedObject(getMemoryForIndex(index))}
        onRowClick={({ event, index, rowData }) => {
          console.log("Clicked ", event, index, rowData);
          if (rowData && rowData.uuid) {
            onShowMemeoryDetail(rowData.uuid);
          }
        }}
        columns={[
          {
            width: 60,
            label: "ID",
            dataKey: "id",
            numeric: true,
          },
          {
            width: 120,
            label: "Type",
            dataKey: "type",
            numeric: false,
          },
          {
            width: 120,
            label: "Name",
            dataKey: "name",
            numeric: false,
          },
        ]}
      />
    </Paper>
  );
}
