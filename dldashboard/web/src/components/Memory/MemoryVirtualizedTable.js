import React from "react";
import PropTypes from "prop-types";
import clsx from "clsx";
import { withStyles } from "@material-ui/core/styles";
import TableCell from "@material-ui/core/TableCell";
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
      backgroundColor: theme.palette.action.hover,
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
    rowHeight: 36,
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

    return (
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
  return {
    uuid: arr.data[0],
    id: arr.data[1],
    type: arr.data[9],
    name: arr.data[7],
  };
}

/**
 * Creates a virualized table for the objects in Memory.
 *
 * See Also: https://material-ui.com/components/tables/#virtualized-table
 *
 * @param {memoryManager, onShowMemoryDetail } props
 *                            memoryManager @see MemoryManager
 *                            onShowMemoryDetail event handler for memory detail selection.
 */
export default function ReactVirtualizedTable({
  memoryManager,
  onShowMemoryDetail,
}) {
  const [getCount, getMemoryForIndex] = memoryManager;

  return (
    <AutoSizer>
      {({ height, width }) => (
        <VirtualizedTable
          height={height}
          width={width}
          rowCount={getCount()}
          rowGetter={({ index }) =>
            toReferencedObject(getMemoryForIndex(index))
          }
          onRowClick={({ event, index, rowData }) => {
            if (rowData && rowData.uuid) {
              onShowMemoryDetail(rowData.uuid);
            }
          }}
          columns={[
            {
              flexGrow: 1,
              width: 1,
              label: "ID",
              dataKey: "id",
              numeric: true,
            },
            {
              flexGrow: 5,
              width: 5,
              label: "Type",
              dataKey: "type",
              numeric: false,
            },
            {
              flexGrow: 5,
              width: 5,
              label: "Name",
              dataKey: "name",
              numeric: false,
            },
          ]}
        />
      )}
    </AutoSizer>
  );
}
