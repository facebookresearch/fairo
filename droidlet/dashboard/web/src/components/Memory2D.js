/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D.js

import React from "react";
import { Stage, Layer, Circle, Line, Text, Group } from "react-konva";
import { schemeCategory10 as colorScheme } from "d3-scale-chromatic";
import MemoryMapTable from "./Memory2D/MemoryMapTable";

var hashCode = function (s) {
  return s.split("").reduce(function (a, b) {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);
};

const DEFAULT_SPACING = 12;

class Memory2D extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      height: 400,
      width: 600,
      isLoaded: false,
      memory: null,
      detections_from_memory: [],
      xmin: -10,
      xmax: 10,
      ymin: -10,
      ymax: 10,
      bot_xyz: [0.0, 0.0, 0.0],
      bot_data: null,
      tooltip: null,
      stageScale: 1,
      stageX: 0,
      stageY: 0,
      memory2d_className: "memory2d",
      drag_coordinates: [0, 0],
      enlarge_bot_marker: false,
      table_data: null,
      table_visible: false,
      table_coords: [0, 0],
    };
    this.state = this.initialState;
    this.outer_div = React.createRef();
    this.resizeHandler = this.resizeHandler.bind(this);
    this.handleObjClick = this.handleObjClick.bind(this);
  }
  handleDrag = (className, drag_coordinates) => {
    this.setState({ memory2d_className: className });
    if (drag_coordinates) {
      this.setState({ drag_coordinates });
    }
  };
  convertGridCoordinate = (xy) => {
    const { xmax, xmin, ymax, ymin } = this.state;
    let { width, height } = this.state;
    width = Math.min(width, height);
    height = width;
    return [
      (xy[1] * (ymax - ymin)) / height + ymin,
      0,
      (xy[0] * (xmax - xmin)) / width + xmin,
    ];
  };
  convertCoordinate = (xyz) => {
    const { xmax, xmin, ymax, ymin } = this.state;
    let { width, height } = this.state;
    width = Math.min(width, height);
    height = width;
    let x = parseInt(((xyz[2] - xmin) / (xmax - xmin)) * width);
    let y = parseInt(((-xyz[0] - ymin) / (ymax - ymin)) * height);
    y = height - y;
    return [x, y];
  };
  /* zoom-in, zoom-out function
     the maximum zoom-out is 100% (stageScale = 1), the maximum zoom-in is unlimited 
  */
  handleWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.2;
    const stage = e.target.getStage();
    const oldScale = stage.scaleX();
    const mousePointTo = {
      x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
      y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
    };

    const tmpScale = e.evt.deltaY > 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const newScale = tmpScale < 1 ? 1 : tmpScale;

    this.setState({
      drag_coordinates: [
        -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
        -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale,
      ],
      stageScale: newScale,
      stageX:
        -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
      stageY:
        -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale,
    });
  };
  handleObjClick = (obj_type, x, y, data) => {
    /* 
     * Potentially useful code to debug table positioning when zooming in *
     *
     * 
    let {
      table_coords,
      drag_coordinates,
      stageX,
      stageY,
      stageScale,
      height,
      width,
    } = this.state;
    width = Math.min(width, height);
    height = width;

    console.log("x:", x, "y:", y);
    console.log("drag_coordinates:", drag_coordinates[0], ", ", drag_coordinates[1]);
    console.log("stage_coords:", stageX, ", ", stageY);
    console.log("stageScale:", stageScale);
    console.log("width:", width, "height:", height);
    console.log("will plot table at: [", (x + drag_coordinates[0]), ", ", (Math.min(height, width) - y - drag_coordinates[1]), "]");
    */

    console.log(obj_type + " clicked");

    this.setState({
      table_visible: true,
      table_coords: [x, y],
      table_data: data,
    });
  };
  positionTable = (h, w, tc, dc, td) => {
    // this takes all these parameters so table will properly update position on change
    let ret = { position: "absolute" };
    let final_coords = [tc[0] + dc[0], Math.min(h, w) - (tc[1] + dc[1])];
    let final_pos = ["left", "bottom"];
    let table_dims = [200, 61 * Object.keys(td).length + 100];
    if (final_coords[1] > Math.min(h, w) - table_dims[1]) {
      final_coords[1] = Math.min(h, w) - final_coords[1];
      final_pos[1] = "top";
    }
    ret[final_pos[0]] = final_coords[0];
    ret[final_pos[1]] = final_coords[1];
    return ret;
  };
  resizeHandler() {
    if (this.props.isMobile) {
      let dimensions = this.props.dimensions;
      if (
        (dimensions !== undefined && dimensions !== this.state.height) ||
        (dimensions !== undefined && dimensions !== this.state.width)
      ) {
        this.setState({ height: dimensions, width: dimensions });
      }
    } else {
      if (this.outer_div != null && this.outer_div.current != null) {
        let { clientHeight, clientWidth } = this.outer_div.current;
        if (
          (clientHeight !== undefined && clientHeight !== this.state.height) ||
          (clientWidth !== undefined && clientWidth !== this.state.width)
        ) {
          this.setState({ height: clientHeight, width: clientWidth });
        }
      }
    }
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    if (this.props.glContainer !== undefined) {
      // if this is inside a golden-layout container
      this.props.glContainer.on("resize", this.resizeHandler);
    }
  }
  componentDidUpdate(prevProps, prevState) {
    const { bot_xyz, bot_data } = this.state;
    let { xmin, xmax, ymin, ymax } = this.state;
    let bot_x = bot_xyz[0],
      bot_y = bot_xyz[1];
    let state_changed = false;
    if (bot_x < xmin) {
      xmin = bot_x;
      state_changed = true;
    }
    if (bot_x > xmax) {
      xmax = bot_x;
      state_changed = true;
    }
    if (bot_y < ymin) {
      ymin = bot_y;
      state_changed = true;
    }
    if (bot_y > ymax) {
      ymax = bot_y;
      state_changed = true;
    }
    if (state_changed) {
      this.setState({ xmin: xmin, ymin: ymin, xmax: xmax, ymax: ymax });
    }
    this.resizeHandler();
  }

  componentWillUnmount() {
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
  }

  render() {
    if (!this.state.isLoaded) return <p>Loading</p>;
    let {
      height,
      width,
      memory,
      detections_from_memory,
      bot_data,
      obstacle_map,
      tooltip,
      drag_coordinates,
      stageScale,
      enlarge_bot_marker,
      table_data,
      table_visible,
      table_coords,
    } = this.state;
    width = Math.min(width, height);
    height = width;
    let { objects } = memory;
    let { xmin, xmax, ymin, ymax } = this.state;

    let bot_x = bot_data.pos[2];
    let bot_y = -bot_data.pos[0];
    let bot_yaw = bot_data.yaw;

    if (height === 0 && width === 0) {
      // return early for performance
      return (
        <div ref={this.outer_div} style={{ height: "100%", width: "100%" }} />
      );
    }
    if (
      !this.props.stateManager.dash_enable_map ||
      !this.props.stateManager.agent_enable_map
    ) {
      return <p>Map Disabled</p>;
    }

    bot_x = parseInt(((bot_x - xmin) / (xmax - xmin)) * width);
    bot_y = parseInt(((bot_y - ymin) / (ymax - ymin)) * height);
    bot_y = height - bot_y;

    let renderedObjects = [];
    let mapBoundary = [];
    let j = 0;

    // Visualize map
    if (obstacle_map) {
      obstacle_map.forEach((obj) => {
        let color = "#827f7f";
        let x = parseInt(((obj[0] - xmin) / (xmax - xmin)) * width);
        let y = parseInt(((obj[1] - ymin) / (ymax - ymin)) * height);
        renderedObjects.push(
          // FIXME: may need to also add mobile support? by using onTap/onTouchEnd
          <Circle
            key={j++}
            radius={2}
            x={x}
            y={y}
            fill={color}
            onClick={(e) => {
              this.handleObjClick("obstacle_map", x, y, {
                x: obj[0],
                y: obj[1],
              });
            }}
            onMouseOver={(e) => {
              e.currentTarget.setRadius(5);
            }}
            onMouseOut={(e) => {
              e.currentTarget.setRadius(2);
            }}
          />
        );
      });
    }

    // Put detected objects from memory on map
    detections_from_memory.forEach((obj) => {
      let obj_id = obj.obj_id;
      let xyz = obj.pos;
      let color = "#0000FF";
      let x = parseInt(((xyz[2] - xmin) / (xmax - xmin)) * width);
      let y = parseInt(((-xyz[0] - ymin) / (ymax - ymin)) * height);
      y = height - y;
      renderedObjects.push(
        <Circle
          key={obj.memid}
          radius={3}
          x={x}
          y={y}
          fill={color}
          onClick={(e) => {
            this.handleObjClick("detection_from_memory", x, y, obj);
          }}
          onMouseOver={(e) => {
            e.currentTarget.setRadius(6);
          }}
          onMouseOut={(e) => {
            e.currentTarget.setRadius(3);
          }}
        />
      );
    });

    if (objects !== undefined && objects.forEach !== undefined) {
      objects.forEach((obj, key, map) => {
        let color = colorScheme[Math.abs(hashCode(obj.label)) % 10];
        let xyz = obj.xyz;
        let x = parseInt(((xyz[2] - xmin) / (xmax - xmin)) * width);
        let y = parseInt(((-xyz[0] - ymin) / (ymax - ymin)) * height);
        y = height - y;
        renderedObjects.push(
          <Circle
            key={j++}
            radius={3}
            x={x}
            y={y}
            fill={color}
            onMouseOver={(e) => {
              e.currentTarget.setRadius(6);
            }}
            onMouseOut={(e) => {
              e.currentTarget.setRadius(3);
            }}
          />
        );
        // renderedObjects.push(<Text key={j++} text={obj.label} x={x} y={y} fill={color} fontSize={10} />);
      });
    }

    /* bot marker
      a circle to show position and
      a line to show orientation
    */
    renderedObjects.push(
      <Group
        key={bot_data.memid}
        x={bot_x}
        y={bot_y}
        onClick={(e) => {
          this.handleObjClick("bot", bot_x, bot_y, bot_data);
        }}
        onMouseOver={(e) => {
          this.setState({ enlarge_bot_marker: true });
        }}
        onMouseOut={(e) => {
          this.setState({ enlarge_bot_marker: false });
        }}
      >
        <Circle key={j++} radius={enlarge_bot_marker ? 15 : 10} fill="red" />
        <Line
          key={j++}
          points={enlarge_bot_marker ? [0, 0, 18, 0] : [0, 0, 12, 0]}
          rotation={(-bot_yaw * 180) / Math.PI}
          stroke="black"
          strokeWidth={enlarge_bot_marker ? 1.5 : 1}
        />
      </Group>
    );

    var gridLayer = [];
    var padding = 10;
    var gridKey = 12344;
    for (var i = 0; i < width / padding; i++) {
      gridLayer.push(
        <Line
          key={gridKey + i}
          points={[
            Math.round(i * padding) + 0.5,
            0,
            Math.round(i * padding) + 0.5,
            height,
          ]}
          stroke="#f5f5f5"
          strokeWidth={1}
        />
      );
    }

    //gridLayer.push(<Line key={gridKey + i++} points={[0, 0, 10, 10]} />); // this draws a diagonal line at the top left of the grid?
    for (j = 0; j < height / padding; j++) {
      gridLayer.push(
        <Line
          key={gridKey + i + j}
          points={[0, Math.round(j * padding), width, Math.round(j * padding)]}
          stroke="#ddd"
          strokeWidth={0.5}
        />
      );
    }

    let coordinateAxesLayer = [];
    let rootPointDefault = [9, 0, -9];
    let coordinateRootPoint = this.convertCoordinate(rootPointDefault);

    let x = (coordinateRootPoint[0] - drag_coordinates[0]) / stageScale;
    let y = (coordinateRootPoint[1] - drag_coordinates[1]) / stageScale;

    let rootPoint = this.convertGridCoordinate([x, y]);
    let strokeWidth = 0.5 / stageScale;

    let axesZ = (
      <Line
        key="axesZ"
        points={[0, 0, width, 0]}
        x={x}
        y={y}
        stroke="#AAAAAA"
        strokeWidth={strokeWidth}
      />
    );

    let axesX = (
      <Line
        key="axesX"
        points={[0, -height, 0, 0]}
        x={x}
        y={y}
        stroke="#AAAAAA"
        strokeWidth={strokeWidth}
      />
    );
    /* add notches and coordinate point value to coordinate axes
       the number of notches axes is unchanged when zoom-in, zoom-out 
       the coordinate point values is shown rounded 2 decimal digits 
       (e.g: 5.123 = 5.12; 5.129 = 5.13) 
    */
    let notches = [];
    let endPointXZ = [
      coordinateRootPoint[0] + width,
      coordinateRootPoint[1] - height,
    ];
    let tmpPointX = coordinateRootPoint[0];
    let tmpPointY = coordinateRootPoint[1];

    while (tmpPointX < endPointXZ[0]) {
      tmpPointX += 30;
      let coordinate = this.convertGridCoordinate([
        (tmpPointX - drag_coordinates[0]) / stageScale,
        0,
      ]);
      notches.push(
        <Text
          key={"textCoordinateX-" + tmpPointX}
          text={coordinate[2].toFixed(2)}
          fontSize={10 / stageScale}
          x={(tmpPointX - 10 - drag_coordinates[0]) / stageScale}
          y={(coordinateRootPoint[1] - 15 - drag_coordinates[1]) / stageScale}
          fill="#AAAAAA"
        />
      );
      notches.push(
        <Line
          key={"coordinateX-" + tmpPointX}
          points={[0, -3 / stageScale, 0, 3 / stageScale]}
          x={(tmpPointX - drag_coordinates[0]) / stageScale}
          y={y}
          stroke="#AAAAAA"
          strokeWidth={strokeWidth}
        />
      );
    }

    while (tmpPointY > endPointXZ[1]) {
      tmpPointY = tmpPointY - 20;
      let coordinate = this.convertGridCoordinate([
        0,
        (tmpPointY - drag_coordinates[1]) / stageScale,
      ]);
      notches.push(
        <Text
          key={"textCoordinateY-" + tmpPointY}
          text={coordinate[0].toFixed(2)}
          fontSize={10 / stageScale}
          x={(coordinateRootPoint[0] - 35 - drag_coordinates[0]) / stageScale}
          y={(tmpPointY - 5 - drag_coordinates[1]) / stageScale}
          fill="#AAAAAA"
        />
      );
      notches.push(
        <Line
          key={"coordinateY-" + tmpPointY}
          points={[-3 / stageScale, 0, 3 / stageScale, 0]}
          x={x}
          y={(tmpPointY - drag_coordinates[1]) / stageScale}
          stroke="#AAAAAA"
          strokeWidth={strokeWidth}
        />
      );
    }

    let rootTextCoordinate = [
      (coordinateRootPoint[0] - 25 - drag_coordinates[0]) / stageScale,
      (coordinateRootPoint[1] + 10 - drag_coordinates[1]) / stageScale,
    ];
    notches.push(
      <Text
        key="root-text"
        fill="#AAAAAA"
        text={`${rootPoint[0].toFixed(2)}, ${rootPoint[2].toFixed(2)}`}
        fontSize={10 / stageScale}
        x={rootTextCoordinate[0]}
        y={rootTextCoordinate[1]}
      />
    );
    notches.push(
      <Text
        key="x-text"
        fill="#AAAAAA"
        text="x"
        fontSize={12 / stageScale}
        x={
          (this.convertCoordinate([-9, 0, -8.8])[0] - drag_coordinates[0]) /
          stageScale
        }
        y={
          (this.convertCoordinate([-9, 0, -8.8])[1] - drag_coordinates[1]) /
          stageScale
        }
      />
    );
    notches.push(
      <Text
        key="z-text"
        fill="#AAAAAA"
        text="z"
        fontSize={12 / stageScale}
        x={
          (this.convertCoordinate([9.1, 0, 9])[0] - drag_coordinates[0]) /
          stageScale
        }
        y={
          (this.convertCoordinate([9.1, 0, 9])[1] - drag_coordinates[1]) /
          stageScale
        }
      />
    );
    coordinateAxesLayer.push(axesX, axesZ, notches);

    // table props
    const onTableDone = (e) => {
      this.setState({ table_visible: false });
    };
    const rows = [];
    if (table_visible) {
      console.assert(table_data !== null, "table_data should be initialized");
      let data = Object.entries(table_data);
      data.forEach((entry) =>
        rows.push({
          attribute: entry[0].toString(),
          value: entry[1].toString(),
        })
      );
    }

    // final render
    return (
      <div
        ref={this.outer_div}
        style={{ height: "100%", width: "100%", position: "relative" }}
      >
        <div style={{ position: "absolute" }}>
          <Stage
            draggable
            className={this.state.memory2d_className}
            width={width}
            height={height}
            onWheel={this.handleWheel}
            scaleX={this.state.stageScale}
            scaleY={this.state.stageScale}
            x={this.state.stageX}
            y={this.state.stageY}
            onDragMove={(e) =>
              this.handleDrag("memory2d dragging-memory2d", [
                e.target.attrs.x,
                e.target.attrs.y,
              ])
            }
            onDragEnd={(e) =>
              this.handleDrag("memory2d", [e.target.attrs.x, e.target.attrs.y])
            }
          >
            <Layer className="gridLayer">{gridLayer}</Layer>
            <Layer className="coordinateAxesLayer">{coordinateAxesLayer}</Layer>
            <Layer className="mapBoundary">{mapBoundary}</Layer>
            <Layer className="renderedObjects">{renderedObjects}</Layer>
            <Layer>
              <Text
                text={tooltip}
                offsetX={-DEFAULT_SPACING}
                offsetY={-DEFAULT_SPACING}
                visible={tooltip !== null}
                shadowEnabled={true}
              />
            </Layer>
          </Stage>
        </div>
        {table_visible && (
          <div
            style={this.positionTable(
              height,
              width,
              table_coords,
              drag_coordinates,
              table_data
            )}
          >
            <MemoryMapTable rows={rows} onTableDone={onTableDone} />
          </div>
        )}
      </div>
    );
  }
}

export default Memory2D;
