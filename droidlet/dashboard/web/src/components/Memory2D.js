/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D.js

import React from "react";
import { Stage, Layer, Circle, Line, Text, Group } from "react-konva";
import { schemeCategory10 as colorScheme } from "d3-scale-chromatic";
import MemoryMapTable, {
  positionMemoryMapTable,
} from "./Memory2D/MemoryMapTable";
import MemoryPopup, { positionMemoryPopup } from "./Memory2D/MemoryPopup";

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
      popup_data: null,
      popup_visible: false,
      popup_coords: [0, 0],
      map_update_count: 0,
      grouping_mode: false,
      grouped_objects: [],
    };
    this.state = this.initialState;
    this.outer_div = React.createRef();
    this.resizeHandler = this.resizeHandler.bind(this);
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
  handleObjClick = (obj_type, map_pos, data) => {
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

    let { grouping_mode, grouped_objects, popup_coords } = this.state;

    if (!grouping_mode) {
      console.log(obj_type + " clicked reg");
      this.setState({
        table_visible: true,
        table_coords: map_pos,
        table_data: data,
      });

      if (map_pos[0] !== popup_coords[0] || map_pos[1] !== popup_coords[1]) {
        this.setState({ popup_visible: false });
      }
    } else {
      console.log(obj_type + " clicked group");
      this.setState({ grouped_objects: [...grouped_objects, data] });
    }
  };
  handlePopupClick = (map_pos, data) => {
    let { table_coords } = this.state;

    this.setState({
      popup_visible: true,
      popup_coords: map_pos,
      popup_data: data,
    });

    if (map_pos[0] !== table_coords[0] || map_pos[1] !== table_coords[1]) {
      this.setState({ table_visible: false });
    }
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
      popup_data,
      popup_visible,
      popup_coords,
    } = this.state;
    width = Math.min(width, height);
    height = width;
    let { objects } = memory;
    let { xmin, xmax, ymin, ymax } = this.state;

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

    let renderedObjects = [];
    let mapBoundary = [];
    let j = 0;

    let objectPosPool = {};

    // Pool obstacles by position
    if (obstacle_map) {
      obstacle_map.forEach((obj) => {
        let color = "#827f7f";
        let x = parseInt(((obj[0] - xmin) / (xmax - xmin)) * width);
        let y = parseInt(((obj[1] - ymin) / (ymax - ymin)) * height);
        let map_pos = "" + x + "," + y;
        let poolData = {
          type: "obstacle_map",
          radius: 2,
          radiusFocused: 5,
          color: color,
          data: {
            memid: "don't edit" + j++,
            x: obj[0],
            y: obj[1],
            pos: "[" + obj[0] + ",0," + obj[1] + "]",
          },
        };
        if (!(map_pos in objectPosPool)) {
          objectPosPool[map_pos] = [poolData];
        } else {
          objectPosPool[map_pos].push(poolData);
        }
      });
    }

    // Pool detected objects from memory by position
    detections_from_memory.forEach((obj) => {
      let xyz = obj.pos;
      let color = "#0000FF";
      let [map_x, map_y] = this.convertCoordinate(xyz);
      let map_pos = "" + map_x + "," + map_y;
      let poolData = {
        type: "detection_from_memory",
        radius: 6,
        radiusFocused: 9,
        color: color,
        data: obj,
      };
      if (!(map_pos in objectPosPool)) {
        objectPosPool[map_pos] = [poolData];
      } else {
        objectPosPool[map_pos].push(poolData);
      }
    });

    // Put objects from undefined memory on map?
    if (objects !== undefined && objects.forEach !== undefined) {
      objects.forEach((obj, key, map) => {
        let color = colorScheme[Math.abs(hashCode(obj.label)) % 10];
        let xyz = obj.xyz;
        let [map_x, map_y] = this.convertCoordinate(xyz);
        renderedObjects.push(
          <Circle
            key={j++}
            radius={3}
            x={map_x}
            y={map_y}
            fill={color}
            onMouseOver={(e) => {
              e.currentTarget.setRadius(6);
            }}
            onMouseOut={(e) => {
              e.currentTarget.setRadius(3);
            }}
          />
        );
        renderedObjects.push(
          <Text
            key={j++}
            text={obj.label}
            x={x}
            y={y}
            fill={color}
            fontSize={10}
          />
        );
      });
    }

    // Use pooling to plot points/groups on map
    Object.entries(objectPosPool).forEach((entry) => {
      let [map_pos, objs_at_pos] = entry;
      let [map_x, map_y] = map_pos.split(",");
      map_x = parseInt(map_x);
      map_y = parseInt(map_y);
      if (objs_at_pos.length === 1) {
        // only one object at map position
        let obj = objs_at_pos[0];
        renderedObjects.push(
          <Circle
            key={obj.data.memid}
            radius={obj.radius}
            x={map_x}
            y={map_y}
            fill={obj.color}
            onClick={() => {
              this.handleObjClick(obj.type, [map_x, map_y], obj.data);
            }}
            onMouseOver={(e) => {
              e.currentTarget.setRadius(obj.radiusFocused);
            }}
            onMouseOut={(e) => {
              e.currentTarget.setRadius(obj.radius);
            }}
          />
        );
      } else {
        // several objects overlayed at map position
        let numObjs = objs_at_pos.length;
        let overlayedObjects = [];
        let [groupColor, groupRadius] = ["#0000FF", 6];
        objs_at_pos.forEach((obj) => {
          overlayedObjects.push(obj);
        });
        renderedObjects.push(
          <Group
            key={map_pos}
            x={map_x}
            y={map_y}
            onClick={() => {
              this.handlePopupClick([map_x, map_y], overlayedObjects);
            }}
          >
            {/* {overlayedObjects} */}
            <Circle
              x={0}
              y={0}
              radius={groupRadius}
              fill={groupColor}
              stroke="black"
              strokeWidth={1}
              onMouseOver={(e) => {
                e.currentTarget.setRadius(groupRadius * 1.5);
              }}
              onMouseOut={(e) => {
                e.currentTarget.setRadius(groupRadius);
              }}
            />
            <Text
              x={numObjs > 9 ? -5 : -3}
              y={numObjs > 9 ? -5 : -3}
              width={numObjs > 9 ? 10 : 6}
              height={numObjs > 9 ? 10 : 6}
              text={numObjs > 9 ? "9+" : numObjs}
              fontSize={numObjs > 9 ? 8 : 10}
              fontFamily="Segoe UI"
              fill="white"
              align="center"
              verticalAlign="middle"
            />
          </Group>
        );
      }
    });

    /* bot marker
      a circle to show position and
      a line to show orientation
    */
    let [bot_x, bot_y] = this.convertCoordinate(bot_data.pos);
    let bot_yaw = bot_data.yaw;
    renderedObjects.push(
      <Group
        key={bot_data.memid}
        x={bot_x}
        y={bot_y}
        onClick={() => {
          this.handleObjClick("bot", [bot_x, bot_y], bot_data);
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
      // Vertical Lines
      let startX = drag_coordinates[0] % (padding * stageScale);
      let plotX =
        startX +
        Math.round(i * padding * stageScale - drag_coordinates[0]) +
        0.5;
      gridLayer.push(
        <Line
          key={gridKey + i}
          points={[
            plotX / stageScale,
            (0 - drag_coordinates[1]) / stageScale,
            plotX / stageScale,
            (height - drag_coordinates[1]) / stageScale,
          ]}
          stroke="#f5f5f5"
          strokeWidth={1}
        />
      );
    }

    //gridLayer.push(<Line key={gridKey + i++} points={[0, 0, 10, 10]} />); // this draws a diagonal line at the top left of the grid?
    for (j = 0; j < height / padding; j++) {
      // Horizontal Lines
      let startY = drag_coordinates[1] % (padding * stageScale);
      let plotY =
        startY + Math.round(j * padding * stageScale - drag_coordinates[1]);
      gridLayer.push(
        <Line
          key={gridKey + i + j}
          points={[
            (0 - drag_coordinates[0]) / stageScale,
            plotY / stageScale,
            (width - drag_coordinates[0]) / stageScale,
            plotY / stageScale,
          ]}
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
    const onTableClose = () => {
      this.setState({ table_visible: false });
    };
    const onTableSubmit = (em) => {
      let numChanged = Object.keys(em).reduce((numChanged, attr) => {
        if (em[attr].status === "changed") numChanged += 1;
        return numChanged;
      }, 0);
      if (numChanged > 0) this.props.stateManager.sendManualEdits(em);
      this.setState({ table_visible: false });
    };
    const onPopupClose = () => {
      this.setState({ popup_visible: false, table_visible: false });
    };

    // final render
    return (
      <div
        ref={this.outer_div}
        style={{ height: "100%", width: "100%", position: "relative" }}
        onKeyDown={(e) => {
          // console.log(e.key, "down");
          let selectionKeys = ["Meta", "Command", "Ctrl"];
          if (selectionKeys.includes(e.key)) {
            this.setState({ grouping_mode: true });
          }
        }}
        onKeyUp={() => {
          this.setState({ grouping_mode: false });
        }}
        tabIndex="0"
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
            <Layer
              className="renderedObjects"
              key={this.state.map_update_count}
            >
              {renderedObjects}
            </Layer>
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
        {popup_visible && (
          <div
            style={positionMemoryPopup(
              this.state.height,
              this.state.width,
              popup_coords,
              drag_coordinates,
              popup_data
            )}
          >
            <MemoryPopup
              data={popup_data}
              map_pos={popup_coords}
              onPopupClose={onPopupClose}
              handleObjClick={this.handleObjClick}
            />
          </div>
        )}
        {table_visible && (
          <div
            style={positionMemoryMapTable(
              this.state.height,
              this.state.width,
              table_coords,
              drag_coordinates,
              table_data
            )}
          >
            <MemoryMapTable
              data={table_data}
              onTableClose={onTableClose}
              onTableSubmit={onTableSubmit}
            />
          </div>
        )}
      </div>
    );
  }
}

export default Memory2D;
