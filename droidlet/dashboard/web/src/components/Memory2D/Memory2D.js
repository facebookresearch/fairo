/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/Memory2D/Memory2D.js

import React from "react";

import * as M2DC from "./Memory2DConstants";

import { Stage, Layer, Circle, Line, Text, Group, Shape } from "react-konva";
import { schemeCategory10 as colorScheme } from "d3-scale-chromatic";
import MemoryMapTable, { positionMemoryMapTable } from "./MemoryMapTable";
import ClusteredObjsPopup, {
  positionClusteredObjsPopup,
} from "./ClusteredObjsPopup";
import Memory2DMenu from "./Memory2DMenu";
import IconButton from "@material-ui/core/IconButton";
import MenuIcon from "@material-ui/icons/Menu";

var hashCode = function (s) {
  return s.split("").reduce(function (a, b) {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);
};

class Memory2D extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      height: M2DC.INITIAL_HEIGHT,
      width: M2DC.INITIAL_WIDTH,
      isLoaded: false,
      memory: null,
      triples: null,
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
      focused_point_coords: [null, null],
      table_data: null,
      table_visible: false,
      popup_data: null,
      popup_visible: false,
      map_update_count: 0,
      selection_mode: false,
      drawing_mode: false,
      draw_pos_curr: null,
      draw_pos_start: null,
      selected_objects: {},
      group_name: "",
      grouping_count: 0,
      showMenu: false,
      dynamicPositioning: false,
      showTriples: false,
      mapView: "ZX",
      squareMap: false,
      nodeColorings: M2DC.DEFAULT_NODE_COLORINGS,
    };
    this.state = this.initialState;
    this.outer_div = React.createRef();
    this.stage_ref = React.createRef();
    this.resizeHandler = this.resizeHandler.bind(this);
  }

  /*########################
  ####  Stage Handlers  ####
  ########################*/
  handleDrag = (className, drag_coordinates) => {
    this.setState({ memory2d_className: className });
    if (drag_coordinates) {
      this.setState({ drag_coordinates });
    }
  };
  /* zoom-in, zoom-out function
     the maximum zoom-out is 100% (stageScale = 1), the maximum zoom-in is unlimited 
  */
  handleWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = M2DC.SCALE_FACTOR;
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

  /*###############################
  ####  Coordinate Conversion  ####
  ###############################*/
  /**
   * Takes the position of an object on the grid
   * and converts it to droidlet coords
   * (NOTE: output is 2d)
   */
  convertGridCoordinate = (gridHorz, gridVert) => {
    const { xmax, xmin, ymax, ymin } = this.state;
    let { width, height } = this.state;

    // Maintain 1:1 aspect ratio by treating grid as square at all times
    width = Math.min(width, height);
    height = width;
    let gridLength = Math.min(xmax - xmin, ymax - ymin);

    return [
      (gridHorz * gridLength) / width + xmin,
      (gridVert * gridLength) / height + ymin,
    ];
  };
  /**
   * Takes droidlet coords of object and converts
   * to 2d grid position (based on desired mapView)
   */
  convertCoordinate = (xyz) => {
    const { xmax, xmin, ymax, ymin } = this.state;
    let { width, height, mapView } = this.state;

    // Maintain 1:1 aspect ratio by treating grid as square at all times
    width = Math.min(width, height);
    height = width;
    let gridLength = Math.min(xmax - xmin, ymax - ymin);

    let horz, vert;
    switch (mapView) {
      case "ZX":
        [horz, vert] = [xyz[2], -xyz[0]];
        break;
      case "XY":
        [horz, vert] = [xyz[0], -xyz[1]];
        break;
      case "YZ":
        [horz, vert] = [xyz[1], -xyz[2]];
        break;
      default:
        console.log("invalid view");
        return [0, 0];
    }
    horz = Math.round(horz);
    vert = Math.round(vert);

    let gridHorz = parseInt(((horz - xmin) / gridLength) * width);
    let gridVert = parseInt(((vert - ymin) / gridLength) * height);
    gridVert = height - gridVert;
    return [gridHorz, gridVert];
  };

  /*####################################
  ####  Tabular Component Handlers  ####
  ####################################*/
  handleObjClick = (obj_type, map_pos, obj_data) => {
    let { selection_mode, selected_objects, focused_point_coords } = this.state;

    // do not interact with currently bugged objects without valid memid's
    if (!["bot", "detection_from_memory"].includes(obj_type)) return;

    if (!selection_mode) {
      // if not in grouping mode, open MemoryMapTable
      this.setState({
        table_data: obj_data,
        table_visible: true,
        focused_point_coords: map_pos,
      });
      // make other tabular elements invisible when switching to new obj/cluster
      if (
        map_pos[0] !== focused_point_coords[0] ||
        map_pos[1] !== focused_point_coords[1]
      ) {
        this.setState({ popup_visible: false });
      }
    } else {
      // otherwise if in selection_mode..
      if (!(obj_data.memid in selected_objects)) {
        // ..select object
        this.setState({
          selected_objects: {
            ...selected_objects,
            [obj_data.memid]: obj_data,
          },
        });
      } else {
        // ..unselect object
        let { [obj_data.memid]: _, ...rest } = selected_objects;
        this.setState({ selected_objects: rest });
      }
    }
  };
  handlePopupClick = (map_pos, clusteredObjects) => {
    let { selection_mode, selected_objects, focused_point_coords } = this.state;

    if (!selection_mode) {
      // if not in selection_mode, open ClusteredObjsPopup
      this.setState({
        popup_data: clusteredObjects,
        popup_visible: true,
        focused_point_coords: map_pos,
      });
      // make other tabular elements invisible when switching to new focus point
      if (
        map_pos[0] !== focused_point_coords[0] ||
        map_pos[1] !== focused_point_coords[1]
      ) {
        this.setState({ table_visible: false });
      }
    } else {
      // otherwise if in selection_mode..
      let toSelect = {};
      let toUnselectFrom = selected_objects;
      clusteredObjects.forEach((obj) => {
        let obj_data = obj.data;
        if (!(obj_data.memid in selected_objects)) {
          // ..select object
          toSelect[obj_data.memid] = obj_data;
        } else {
          // ..unselect object
          let { [obj_data.memid]: _, ...rest } = toUnselectFrom;
          toUnselectFrom = rest;
        }
      });
      this.setState({
        selected_objects: {
          ...toUnselectFrom,
          ...toSelect,
        },
      });
    }
  };
  onTableClose = () => {
    // make table invisible; unfocus point if popup not open
    this.setState((prevState) => {
      return {
        table_visible: false,
        ...(!prevState.popup_visible && { focused_point_coords: [null, null] }),
      };
    });
  };
  onTableSubmit = (em) => {
    let numChanged = Object.keys(em).reduce((numChanged, attr) => {
      if (em[attr].status === "changed") numChanged += 1;
      return numChanged;
    }, 0);
    if (numChanged > 0)
      this.props.stateManager.sendManualChange({
        type: "edit",
        data: em,
      });
    this.onTableClose();
  };
  onTableRestore = (memid) => {
    this.props.stateManager.sendManualChange({
      type: "restore",
      data: memid,
    });
    this.onTableClose();
  };
  onPopupClose = () => {
    this.setState({
      popup_visible: false,
      focused_point_coords: [null, null],
    });
    this.onTableClose();
  };

  /*##########################
  ####  Drawing Handlers  ####
  ##########################*/
  handleToggleDraw = (e) => {
    let { drag_coordinates, stageScale, drawing_mode } = this.state;

    e.evt.preventDefault();
    // close non-stage components
    this.onTableClose();
    this.onPopupClose();

    let draw_pos = {
      x:
        (e.target.getStage().getPointerPosition().x - drag_coordinates[0]) /
        stageScale,
      y:
        (e.target.getStage().getPointerPosition().y - drag_coordinates[1]) /
        stageScale,
    };
    if (!drawing_mode) {
      // start drawing
      this.setState({ draw_pos_start: draw_pos });
    } else {
      // end drawing
      this.handleDrawEnd(draw_pos);
    }
    this.setState({ drawing_mode: !drawing_mode });
  };
  inDrawnBounds = (map_pos, whileDrawing = true, end_pos = null) => {
    let { drawing_mode, draw_pos_curr, draw_pos_start } = this.state;
    let [map_x, map_y] = map_pos;

    let startRect = draw_pos_start;
    let endRect = end_pos ? end_pos : draw_pos_curr;

    if (!(endRect && startRect)) return false;
    let start = {
      x: Math.min(startRect.x, endRect.x),
      y: Math.min(startRect.y, endRect.y),
    };
    let end = {
      x: Math.max(startRect.x, endRect.x),
      y: Math.max(startRect.y, endRect.y),
    };
    return (
      whileDrawing === drawing_mode &&
      map_x > start.x &&
      map_y > start.y &&
      map_x < end.x &&
      map_y < end.y
    );
  };
  handleDrawEnd = (end_pos) => {
    let {
      bot_data,
      obstacle_map,
      detections_from_memory,
      xmin,
      xmax,
      ymin,
      ymax,
      width,
      height,
      selected_objects,
    } = this.state;

    let toSelect = {};
    let toUnselectFrom = selected_objects;

    // Select/unselect map obstacles
    // FIXME: map obstacles do not currently have memid from agent,
    //        disabling grouping for now
    let enableObstacleMapGrouping = false;
    if (enableObstacleMapGrouping && obstacle_map) {
      obstacle_map.forEach((obj) => {
        let map_x = parseInt(((obj[0] - xmin) / (xmax - xmin)) * width);
        let map_y = parseInt(((obj[1] - ymin) / (ymax - ymin)) * height);
        let map_pos = "" + map_x + "," + map_y;
        let data = {
          memid: "don't edit " + map_pos,
          x: obj[0],
          y: obj[1],
          pos: "[" + obj[0] + ",0," + obj[1] + "]",
        };

        if (this.inDrawnBounds([map_x, map_y], true, end_pos)) {
          if (!(data.memid in selected_objects)) {
            toSelect[data.memid] = data;
          } else {
            let { [data.memid]: _, ...rest } = toUnselectFrom;
            toUnselectFrom = rest;
          }
        }
      });
    }

    // Select/unselect bot
    if (
      this.inDrawnBounds(this.convertCoordinate(bot_data.pos), true, end_pos)
    ) {
      if (!(bot_data.memid in selected_objects)) {
        toSelect[bot_data.memid] = bot_data;
      } else {
        let { [bot_data.memid]: _, ...rest } = toUnselectFrom;
        toUnselectFrom = rest;
      }
    }

    // Select/unselect detected objects from memory
    detections_from_memory.forEach((obj) => {
      let xyz = obj.pos;
      let [map_x, map_y] = this.convertCoordinate(xyz);

      if (this.inDrawnBounds([map_x, map_y], true, end_pos)) {
        if (!(obj.memid in selected_objects)) {
          toSelect[obj.memid] = obj;
        } else {
          let { [obj.memid]: _, ...rest } = toUnselectFrom;
          toUnselectFrom = rest;
        }
      }
    });

    this.setState({
      selected_objects: {
        ...toUnselectFrom,
        ...toSelect,
      },
    });
  };
  handleMouseMove = (e) => {
    let { drag_coordinates, stageScale } = this.state;

    let draw_pos = {
      x:
        (e.target.getStage().getPointerPosition().x - drag_coordinates[0]) /
        stageScale,
      y:
        (e.target.getStage().getPointerPosition().y - drag_coordinates[1]) /
        stageScale,
    };
    this.setState({
      draw_pos_curr: draw_pos,
    });
  };

  /*#######################
  ####  Menu Handlers  ####
  #######################*/
  onMenuOpen = () => {
    this.setState({ showMenu: true });
  };
  onGroupSubmit = (data) => {
    this.props.stateManager.sendManualChange({
      type: "group",
      data: data,
    });
    // flush previously selected objects
    this.setState({ selected_objects: {} });
  };
  centerToBot = () => {
    let { height, width, bot_data, squareMap } = this.state;
    if (squareMap) {
      width = Math.min(width, height);
      height = width;
    }
    let [bot_horz, bot_vert] = this.convertCoordinate(bot_data.pos);
    this.setState(
      {
        drag_coordinates: [width / 2 - bot_horz, height / 2 - bot_vert],
        stageScale: 1,
      },
      () => {
        this.stage_ref.current.setAttrs({
          x: width / 2 - bot_horz,
          y: height / 2 - bot_vert,
        });
      }
    );
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
      popup_data,
      popup_visible,
      dynamicPositioning,
      showTriples,
      focused_point_coords,
      map_update_count,
      selected_objects,
      draw_pos_start,
      draw_pos_curr,
      drawing_mode,
      mapView,
      squareMap,
      nodeColorings,
    } = this.state;
    if (squareMap) {
      width = Math.min(width, height);
      height = width;
    }
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
    let j = 0;

    let objectPosPool = {};
    let nodeTypeInfo = {};

    /*#################
    ####  Pooling  ####
    #################*/
    // Pool map obstacles by position
    if (obstacle_map) {
      obstacle_map.forEach((obj) => {
        let color = "#827f7f";
        let map_x = parseInt(((obj[0] - xmin) / (xmax - xmin)) * width);
        let map_y = parseInt(((obj[1] - ymin) / (ymax - ymin)) * height);
        let map_pos = "" + map_x + "," + map_y;
        let poolData = {
          type: "obstacle_map",
          radius: M2DC.OBSTACLE_MAP_RADIUS,
          color: color,
          data: {
            memid: "don't edit " + map_pos,
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
      let nodeType = obj.node_type;
      let color =
        nodeType && nodeColorings[nodeType]
          ? nodeColorings[nodeType]
          : "0000FF";
      let [map_x, map_y] = this.convertCoordinate(xyz);
      let map_pos = "" + map_x + "," + map_y;
      let poolData = {
        type: "detection_from_memory",
        radius: M2DC.DETECTION_FROM_MEMORY_RADIUS,
        color: color,
        data: obj,
      };
      if (!(map_pos in objectPosPool)) {
        objectPosPool[map_pos] = [poolData];
      } else {
        objectPosPool[map_pos].push(poolData);
      }

      if (nodeType) {
        if (!(nodeType in nodeTypeInfo)) {
          nodeTypeInfo[nodeType] = {
            count: 1,
            color: color,
          };
        } else {
          let { count, ...rest } = nodeTypeInfo[nodeType];
          nodeTypeInfo[nodeType] = {
            count: count + 1,
            ...rest,
          };
        }
      }
    });

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
            perfectDrawEnabled={false}
          />
        );
        renderedObjects.push(
          <Text
            key={j++}
            text={obj.label}
            x={map_x}
            y={map_y}
            fill={color}
            fontSize={10}
            perfectDrawEnabled={false}
          />
        );
      });
    }

    /*#########################
    ####  Plotting Points  ####
    #########################*/
    // Use pooling to plot points/groups on map
    Object.entries(objectPosPool).forEach((entry) => {
      let [map_pos, objs_at_pos] = entry;
      let [map_x, map_y] = map_pos.split(",");
      map_x = parseInt(map_x);
      map_y = parseInt(map_y);
      let isFocused =
        this.inDrawnBounds([map_x, map_y]) ||
        (map_x === focused_point_coords[0] &&
          map_y === focused_point_coords[1]);
      if (objs_at_pos.length === 1) {
        // only one object at map position
        let obj = objs_at_pos[0];
        renderedObjects.push(
          <Circle
            key={obj.data.memid}
            x={map_x}
            y={map_y}
            radius={isFocused ? obj.radius * 1.5 : obj.radius}
            fill={obj.data.memid in selected_objects ? "green" : obj.color}
            onClick={(e) => {
              if (e.evt.button === 0)
                this.handleObjClick(obj.type, [map_x, map_y], obj.data);
            }}
            perfectDrawEnabled={false}
          />
        );
      } else {
        // several objects clustered at map position
        let numObjs = objs_at_pos.length;
        let typesOfObjs = new Set();
        let clusteredObjects = [];
        let clusterRadius = M2DC.CLUSTER_RADIUS;
        objs_at_pos.forEach((obj) => {
          if (obj.data.node_type) typesOfObjs.add(obj.data.node_type);
          clusteredObjects.push(obj);
        });
        let someObjSelected = objs_at_pos.some(
          (obj) => obj.data.memid in selected_objects
        );
        let radialFill = [];
        Array.from(typesOfObjs).forEach((type, index) => {
          radialFill.push(
            (index / Math.max(1, typesOfObjs.size - 1)).toFixed(2)
          );
          radialFill.push(nodeColorings[type]);
        });

        renderedObjects.push(
          <Group
            key={map_pos}
            x={map_x}
            y={map_y}
            onClick={(e) => {
              if (e.evt.button === 0)
                this.handlePopupClick([map_x, map_y], clusteredObjects);
            }}
          >
            <Circle
              x={0}
              y={0}
              radius={isFocused ? clusterRadius * 1.5 : clusterRadius}
              fillRadialGradientColorStops={
                someObjSelected ? [0, "green"] : radialFill
              }
              fillRadialGradientStartPoint={{ x: 0, y: 0 }}
              fillRadialGradientStartRadius={0}
              fillRadialGradientEndPoint={{ x: 0, y: 0 }}
              fillRadialGradientEndRadius={
                isFocused ? clusterRadius * 1.5 : clusterRadius
              }
              stroke="black"
              strokeWidth={1}
              perfectDrawEnabled={false}
            />
            <Text
              x={
                numObjs > 9
                  ? clusterRadius * (-5 / 6)
                  : clusterRadius * (-1 / 2)
              }
              y={
                numObjs > 9
                  ? clusterRadius * (-5 / 6)
                  : clusterRadius * (-1 / 2)
              }
              width={numObjs > 9 ? clusterRadius * (5 / 3) : clusterRadius}
              height={numObjs > 9 ? clusterRadius * (5 / 3) : clusterRadius}
              text={numObjs > 9 ? "9+" : numObjs}
              fontSize={numObjs > 9 ? 8 : 10}
              fontFamily={M2DC.FONT}
              fill="black"
              align="center"
              verticalAlign="middle"
              perfectDrawEnabled={false}
            />
          </Group>
        );
      }
    });

    /*####################
    ####  Bot Marker  ####
    ####################*/
    /*
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
        onClick={(e) => {
          if (e.evt.button === 0)
            this.handleObjClick("bot", [bot_x, bot_y], bot_data);
        }}
        onMouseOver={(e) => {
          this.setState({ enlarge_bot_marker: true });
        }}
        onMouseOut={(e) => {
          this.setState({ enlarge_bot_marker: false });
        }}
      >
        <Circle
          key={j++}
          radius={
            enlarge_bot_marker ||
            this.inDrawnBounds([bot_x, bot_y]) ||
            (bot_x === focused_point_coords[0] &&
              bot_y === focused_point_coords[1])
              ? M2DC.BOT_RADIUS * 1.5
              : M2DC.BOT_RADIUS
          }
          fill={bot_data.memid in selected_objects ? "green" : "red"}
          perfectDrawEnabled={false}
        />
        <Line
          key={j++}
          points={
            enlarge_bot_marker ||
            this.inDrawnBounds([bot_x, bot_y]) ||
            (bot_x === focused_point_coords[0] &&
              bot_y === focused_point_coords[1])
              ? [0, 0, 18, 0]
              : [0, 0, 12, 0]
          }
          rotation={(-bot_yaw * 180) / Math.PI}
          stroke="black"
          strokeWidth={
            enlarge_bot_marker ||
            this.inDrawnBounds([bot_x, bot_y]) ||
            (bot_x === focused_point_coords[0] &&
              bot_y === focused_point_coords[1])
              ? 1.5
              : 1
          }
          perfectDrawEnabled={false}
        />
      </Group>
    );

    /*##############
    ####  Grid  ####
    ##############*/
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

    /*##############
    ####  Axes  ####
    ##############*/
    let coordinateAxesLayer = [];

    // Origin of axes fixed at (40, 25) pixels from bottom, left
    let coordinateRootPoint = [M2DC.ROOT_POS[0], height - M2DC.ROOT_POS[1]];

    let rootHorz = (coordinateRootPoint[0] - drag_coordinates[0]) / stageScale;
    let rootVert = (coordinateRootPoint[1] - drag_coordinates[1]) / stageScale;
    let rootPoint = this.convertGridCoordinate(rootHorz, rootVert);

    let axisStrokeWidth = 0.5 / stageScale;

    let axesHorz = (
      <Line
        key="axesHorz"
        points={[
          0,
          0,
          (width - coordinateRootPoint[0] - M2DC.AXES_MARGIN) / stageScale,
          0,
        ]}
        x={rootHorz}
        y={rootVert}
        stroke="#AAAAAA"
        strokeWidth={axisStrokeWidth}
      />
    );

    let axesVert = (
      <Line
        key="axesVert"
        points={[
          0,
          (M2DC.AXES_MARGIN - coordinateRootPoint[1]) / stageScale,
          0,
          0,
        ]}
        x={rootHorz}
        y={rootVert}
        stroke="#AAAAAA"
        strokeWidth={axisStrokeWidth}
      />
    );

    /*#################
    ####  Notches  ####
    #################*/
    /* add notches and coordinate point value to coordinate axes
       the number of notches axes is unchanged when zoom-in, zoom-out 
       the coordinate point values is shown rounded 2 decimal digits 
       (e.g: 5.123 = 5.12; 5.129 = 5.13) 
    */
    let notches = [];
    let tmpPointHorz = coordinateRootPoint[0];
    let tmpPointVert = coordinateRootPoint[1];

    // Notches for horizontal axis
    while (tmpPointHorz < width - (M2DC.AXES_MARGIN + M2DC.NOTCH_SPACING[0])) {
      tmpPointHorz += M2DC.NOTCH_SPACING[0];
      let coordinate = this.convertGridCoordinate(
        (tmpPointHorz - drag_coordinates[0]) / stageScale,
        0
      );
      notches.push(
        <Text
          key={"textCoordinateX-" + tmpPointHorz}
          text={coordinate[0].toFixed(2)}
          fontSize={10 / stageScale}
          x={
            (tmpPointHorz -
              M2DC.HORZ_NOTCH_TEXT_OFFSET[0] -
              drag_coordinates[0]) /
            stageScale
          }
          y={
            (coordinateRootPoint[1] -
              M2DC.HORZ_NOTCH_TEXT_OFFSET[1] -
              drag_coordinates[1]) /
            stageScale
          }
          fill="#AAAAAA"
        />
      );
      notches.push(
        <Line
          key={"coordinateX-" + tmpPointHorz}
          points={[
            0,
            -(M2DC.NOTCH_LENGTH / 2) / stageScale,
            0,
            M2DC.NOTCH_LENGTH / 2 / stageScale,
          ]}
          x={(tmpPointHorz - drag_coordinates[0]) / stageScale}
          y={rootVert}
          stroke="#AAAAAA"
          strokeWidth={axisStrokeWidth}
        />
      );
    }

    // Notches for vertical axis
    while (tmpPointVert > M2DC.AXES_MARGIN + M2DC.NOTCH_SPACING[1]) {
      tmpPointVert -= M2DC.NOTCH_SPACING[1];
      let coordinate = this.convertGridCoordinate(
        0,
        (tmpPointVert - drag_coordinates[1]) / stageScale
      );
      notches.push(
        <Text
          key={"textCoordinateY-" + tmpPointVert}
          text={coordinate[1].toFixed(2)}
          fontSize={10 / stageScale}
          x={
            (coordinateRootPoint[0] -
              M2DC.VERT_NOTCH_TEXT_OFFSET[0] -
              drag_coordinates[0]) /
            stageScale
          }
          y={
            (tmpPointVert -
              M2DC.VERT_NOTCH_TEXT_OFFSET[1] -
              drag_coordinates[1]) /
            stageScale
          }
          fill="#AAAAAA"
        />
      );
      notches.push(
        <Line
          key={"coordinateY-" + tmpPointVert}
          points={[
            -(M2DC.NOTCH_LENGTH / 2) / stageScale,
            0,
            M2DC.NOTCH_LENGTH / 2 / stageScale,
            0,
          ]}
          x={rootHorz}
          y={(tmpPointVert - drag_coordinates[1]) / stageScale}
          stroke="#AAAAAA"
          strokeWidth={axisStrokeWidth}
        />
      );
    }

    // Root Text
    let rootTextCoordinate = [
      (coordinateRootPoint[0] - 25 - drag_coordinates[0]) / stageScale,
      (coordinateRootPoint[1] + 10 - drag_coordinates[1]) / stageScale,
    ];
    notches.push(
      <Text
        key="root-text"
        fill="#AAAAAA"
        text={`${rootPoint[1].toFixed(2)}, ${rootPoint[0].toFixed(2)}`}
        fontSize={10 / stageScale}
        x={rootTextCoordinate[0]}
        y={rootTextCoordinate[1]}
      />
    );

    // Axis Labels
    let vertLabelWindowPos = [35, 2];
    notches.push(
      <Text
        key="vert-axis-text"
        fill="#AAAAAA"
        text={mapView === "ZX" ? "x" : mapView === "XY" ? "y" : "z"}
        fontSize={18 / stageScale}
        x={(vertLabelWindowPos[0] - drag_coordinates[0]) / stageScale}
        y={(vertLabelWindowPos[1] - drag_coordinates[1]) / stageScale}
      />
    );

    let horzLabelWindowPos = [width - 15, height - 35];
    notches.push(
      <Text
        key="horz-axis-text"
        fill="#AAAAAA"
        text={mapView === "ZX" ? "z" : mapView === "XY" ? "x" : "y"}
        fontSize={18 / stageScale}
        x={(horzLabelWindowPos[0] - drag_coordinates[0]) / stageScale}
        y={(horzLabelWindowPos[1] - drag_coordinates[1]) / stageScale}
      />
    );

    coordinateAxesLayer.push(axesVert, axesHorz, notches);

    /*######################
    ####  Final Render  ####
    ######################*/
    return (
      <div
        ref={this.outer_div}
        style={{ height: "100%", width: "100%", position: "relative" }}
        onKeyDown={(e) => {
          let selectionKeys = ["Meta", "Command", "Ctrl"];
          if (selectionKeys.includes(e.key)) {
            this.setState({ selection_mode: true });
          }
          let escapeKeys = ["Escape", "Esc"];
          if (escapeKeys.includes(e.key)) {
            this.setState({
              selection_mode: false,
              drawing_mode: false,
              draw_pos_start: null,
            });
            this.onTableClose();
            this.onPopupClose();
          }
        }}
        onKeyUp={() => {
          this.setState({ selection_mode: false });
        }}
        tabIndex="0"
        onMouseDown={(e) => {
          // quick open/close menu with middle button click
          if (e.button === 1) {
            this.setState((prev) => {
              return {
                showMenu: !prev.showMenu,
              };
            });
          }
        }}
      >
        <div
          style={{
            position: "absolute",
            ...(this.state.drawing_mode && { cursor: "crosshair" }),
          }}
        >
          <Stage
            draggable
            className={this.state.memory2d_className}
            ref={this.stage_ref}
            width={width}
            height={height}
            onWheel={this.handleWheel}
            scaleX={stageScale}
            scaleY={stageScale}
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
            onContextMenu={(e) => {
              // toggle draw selection rectangle with right click
              this.handleToggleDraw(e);
            }}
            onMouseMove={(e) => {
              this.handleMouseMove(e);
            }}
          >
            <Layer className="gridLayer">{gridLayer}</Layer>
            <Layer className="coordinateAxesLayer">{coordinateAxesLayer}</Layer>
            <Layer className="renderedObjects" key={map_update_count}>
              {renderedObjects}
            </Layer>
            <Layer>
              <Text
                text={tooltip}
                offsetX={-M2DC.DEFAULT_SPACING}
                offsetY={-M2DC.DEFAULT_SPACING}
                visible={tooltip !== null}
                shadowEnabled={true}
              />
            </Layer>
            <Layer className="dragToGroup">
              <Shape
                sceneFunc={(ctx, shape) => {
                  if (drawing_mode) {
                    let mouseX = parseInt(draw_pos_curr.x);
                    let mouseY = parseInt(draw_pos_curr.y);
                    let startX = parseInt(draw_pos_start.x);
                    let startY = parseInt(draw_pos_start.y);

                    ctx.clearRect(0, 0, shape.width, shape.height);
                    ctx.beginPath();
                    ctx.rect(startX, startY, mouseX - startX, mouseY - startY);
                    ctx.fillStrokeShape(shape);
                  }
                }}
                fill="transparent"
                stroke="black"
                strokeWidth={4 / stageScale}
                perfectDrawEnabled={false}
              />
            </Layer>
          </Stage>
        </div>
        {popup_visible && (
          <div
            style={positionClusteredObjsPopup(
              this.state.height,
              this.state.width,
              focused_point_coords,
              drag_coordinates,
              dynamicPositioning,
              dynamicPositioning && popup_data
            )}
          >
            <ClusteredObjsPopup
              data={popup_data}
              map_pos={focused_point_coords}
              onPopupClose={this.onPopupClose}
              handleObjClick={this.handleObjClick}
              table_visible={table_visible}
              selection_mode={this.state.selection_mode}
              selected_objects={selected_objects}
            />
          </div>
        )}
        {table_visible && (
          <div
            style={positionMemoryMapTable(
              this.state.height,
              this.state.width,
              focused_point_coords,
              drag_coordinates,
              dynamicPositioning,
              dynamicPositioning && table_data
            )}
          >
            <MemoryMapTable
              data={table_data}
              onTableClose={this.onTableClose}
              onTableSubmit={this.onTableSubmit}
              onTableRestore={this.onTableRestore}
              allTriples={showTriples && this.state.triples}
            />
          </div>
        )}
        <div
          style={{
            position: "absolute",
            right: this.state.width - width,
            top: "-0.1%",
          }}
        >
          <IconButton size="small" disableRipple onClick={this.onMenuOpen}>
            <MenuIcon />
          </IconButton>
        </div>
        <Memory2DMenu
          showMenu={this.state.showMenu}
          onMenuClose={() => {
            this.setState({ showMenu: false });
          }}
          selected_objects={selected_objects}
          onGroupSubmit={this.onGroupSubmit}
          dynamicPositioning={dynamicPositioning}
          toggleDynamicPositioning={() => {
            this.setState((prev) => {
              return {
                dynamicPositioning: !prev.dynamicPositioning,
              };
            });
          }}
          showTriples={showTriples}
          toggleShowTriples={() => {
            this.setState((prev) => {
              return {
                showTriples: !prev.showTriples,
              };
            });
          }}
          mapView={this.state.mapView}
          toggleMapView={(view) => {
            this.setState({ mapView: view }, () => {
              this.centerToBot();
            });
          }}
          centerToBot={this.centerToBot}
          squareMap={squareMap}
          toggleSquareMap={() => {
            this.setState((prev) => {
              return {
                squareMap: !prev.squareMap,
              };
            });
          }}
          nodeTypeInfo={nodeTypeInfo}
          setNodeColoring={(type, color) => {
            this.setState((prev) => {
              return {
                nodeColorings: {
                  ...prev.nodeColorings,
                  [type]: color,
                },
              };
            });
          }}
        />
      </div>
    );
  }
}

export default Memory2D;
