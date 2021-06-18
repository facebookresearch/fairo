/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D.js

import React from "react";
import { Stage, Layer, Circle, Line, Text } from "react-konva";
import { schemeCategory10 as colorScheme } from "d3-scale-chromatic";

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
      xmin: -10,
      xmax: 10,
      ymin: -10,
      ymax: 10,
      bot_xyz: [0.0, 0.0, 0.0],
      tooltip: null,
      stageScale: 1,
      stageX: 0,
      stageY: 0,
      memory2dClassName: "memory2d",
    };
    this.state = this.initialState;
    this.outer_div = React.createRef();
    this.resizeHandler = this.resizeHandler.bind(this);
  }
  handleDrag = (className) => {
    this.setState({ memory2d_className: className });
  };
  convertCoordinate = (xyz) => {
    const { width, height, xmax, xmin, ymax, ymin } = this.state;
    let x = parseInt(((xyz[2] - xmin) / (xmax - xmin)) * width);
    let y = parseInt(((-xyz[0] - ymin) / (ymax - ymin)) * height);
    y = height - y;
    return [x, y];
  };
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
      stageScale: newScale,
      stageX:
        -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
      stageY:
        -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale,
    });
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
    const { bot_xyz } = this.state;
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
    let { height, width, memory, bot_xyz, obstacle_map, tooltip } = this.state;
    let { objects } = memory;
    let { xmin, xmax, ymin, ymax } = this.state;
    let bot_x = bot_xyz[1];
    let bot_y = -bot_xyz[0];
    let bot_yaw = bot_xyz[2];

    if (height === 0 && width === 0) {
      // return early for performance
      return (
        <div ref={this.outer_div} style={{ height: "100%", width: "100%" }} />
      );
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
        mapBoundary.push(
          <Circle key={j++} radius={2} x={x} y={y} fill={color} />
        );
      });
    }

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
          onMouseEnter={(e) => {
            this.setState({
              tooltip: JSON.stringify(obj, null, 4),
            });

            e.currentTarget.setRadius(6);
          }}
          onMouseLeave={(e) => {
            this.setState({ tooltip: null });
            e.currentTarget.setRadius(3);
          }}
        />
      );
      // renderedObjects.push(<Text key={j++} text={obj.label} x={x} y={y} fill={color} fontSize={10} />);
    });

    /* bot marker
      a circle to show position and
      a line to show orientation
    */
    renderedObjects.push(
      <Circle key={j++} radius={10} x={bot_x} y={bot_y} fill="red" />
    );
    renderedObjects.push(
      <Line
        key={j++}
        x={bot_x}
        y={bot_y}
        points={[0, 0, 12, 0]}
        rotation={(-bot_yaw * 180) / Math.PI}
        stroke="black"
      />
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

    gridLayer.push(<Line key={gridKey + i++} points={[0, 0, 10, 10]} />);
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
    let xyz0 = [0, 0, 0];

    let x = this.convertCoordinate(xyz0)[0];
    let y = this.convertCoordinate(xyz0)[1];

    let axesZ = (
      <Line
        key = "axesZ"
        points={[-width / 2, 0, width / 2, 0]}
        x={x}
        y={y}
        stroke="#AAAAAA"
        strokeWidth={0.5}
      />
    );

    let axesX = (
      <Line
        key = "axesX"
        points={[0, -height / 2, 0, height / 2]}
        x={x}
        y={y}
        stroke="#AAAAAA"
        strokeWidth={0.5}
      />
    );
    let notches = [];
    notches.push(
      <Text
        key = "root-text"
        fill="#AAAAAA"
        text={0}
        fontSize={this.state.stageScale > 4 ? 3 : 7}
        x={
          this.state.stageScale > 4
            ? x - 3
            : this.state.stageScale > 2
            ? x - 5
            : x - 10
        }
        y={y + 2}
      />
    );
    for (i = -10; i <= 10; ) {
      let notchesCoordinateX = [i, 0, 0];
      let notchesCoordinateZ = [0, 0, i];
      let tmpCoordinateX = this.convertCoordinate(notchesCoordinateX);
      let tmpCoordinateZ = this.convertCoordinate(notchesCoordinateZ);
      notches.push(
        <Line
          key = {"coordinateX-" + i}
          points={[-3, 0, 3, 0]}
          x={tmpCoordinateX[0]}
          y={tmpCoordinateX[1]}
          stroke="#AAAAAA"
          strokeWidth={0.5}
        />
      );
      notches.push(
        <Line
          key = {"coordinateZ-" + i}
          points={[0, -3, 0, 3]}
          x={tmpCoordinateZ[0]}
          y={tmpCoordinateZ[1]}
          stroke="#AAAAAA"
          strokeWidth={0.5}
        />
      );
      if (i !== 0) {
        notches.push(
          <Text
            key = {"textCoordinateX-" + i}
            fill="#AAAAAA"
            text={i}
            align="right"
            fontSize={this.state.stageScale > 4 ? 3 : 7}
            x={
              this.state.stageScale > 4
                ? tmpCoordinateX[0] - 10
                : tmpCoordinateX[0] - 15
            }
            y={
              this.state.stageScale > 4
                ? tmpCoordinateX[1] - 1.25
                : tmpCoordinateX[1] - 3
            }
          />
        );
        if (i === -0.25) {
          notches.push(
            <Text
              key = {"textCoordinateZ-" + i}
              fill="#AAAAAA"
              text={i}
              fontSize={this.state.stageScale > 4 ? 3 : 7}
              align="center"
              x={tmpCoordinateZ[0] - 7}
              y={tmpCoordinateZ[1] + 4.75}
            />
          );
        } else {
          notches.push(
            <Text
              key = {"textCoordinateZ0-" + i}
              fill="#AAAAAA"
              text={i}
              fontSize={this.state.stageScale > 4 ? 3 : 7}
              align="center"
              x={tmpCoordinateZ[0] - 4}
              y={tmpCoordinateZ[1] + 4.75}
            />
          );
        }
      }
      if (this.state.stageScale > 4) {
        i += 0.25;
      } else if (this.state.stageScale > 2) {
        i += 0.5;
      } else {
        i += 1;
      }
    }
    const textCoordinates = [
      { xy: this.convertCoordinate([-10, 0, 0.2]), label: "-x" },
      { xy: this.convertCoordinate([-0.75, 0, -10]), label: "-z" },
      { xy: this.convertCoordinate([9.5, 0, 0.2]), label: "x" },
      { xy: this.convertCoordinate([-0.75, 0, 10]), label: "z" },
    ];
    textCoordinates.forEach((textCoordinate, key, map) => {
      coordinateAxesLayer.push(
        <Text
          key={textCoordinate.label + key}
          fill="#AAAAAA"
          text={textCoordinate.label}
          x={textCoordinate.xy[0]}
          y={textCoordinate.xy[1]}
        />
      );
    });
    coordinateAxesLayer.push(axesX, axesZ, notches);
    // final render
    return (
      <div ref={this.outer_div} style={{ height: "100%", width: "100%" }}>
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
          onDragMove={() => this.handleDrag("memory2d dragging-memory2d")}
          onDragEnd={() => this.handleDrag("memory2d")}
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
              visible={tooltip != null}
              shadowEnabled={true}
            />
          </Layer>
        </Stage>
      </div>
    );
  }
}

export default Memory2D;
