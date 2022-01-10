/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import stateManager from "../StateManager";
import "status-indicator/styles.css";
import Slider from "rc-slider";
import { Rnd } from "react-rnd";
import "rc-slider/assets/index.css";
import { Stage, Layer, Image as KImage } from "react-konva";

let slider_style = { width: 600, margin: 50 };
const labelStyle = { minWidth: "60px", display: "inline-block" };

/**
 * Currently just a keybinding wrapper that hooks
 * into the state manager and sends events upstream.
 * Later, has to be expanded to visually acknowledge
 * the keys pressed, along with their successful receipt by
 * the backend.
 */
class FrankaArmMover extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      move: 1,
      yaw: 0.01,
      velocity: 1,
      data_logging_time: 30,
      keyboard_enabled: false,
      ee_pos: [0.53, -0.09, 0.43],
      image: null,
    };
    this.update = this.update.bind(this);

    this.state = this.initialState;

    this.handleSubmit = this.handleSubmit.bind(this);
    this.onYawChange = this.onYawChange.bind(this);
    this.onDataLoggingTimeChange = this.onDataLoggingTimeChange.bind(this);
    this.onMoveChange = this.onMoveChange.bind(this);
    this.onVelocityChange = this.onVelocityChange.bind(this);
    this.navRef = React.createRef();
    this.keyCheckRef = React.createRef();
    this.handleClick = this.handleClick.bind(this);
    this.logData = this.logData.bind(this);
    this.stopRobot = this.stopRobot.bind(this);
    this.unstopRobot = this.unstopRobot.bind(this);
    this.keyboardToggle = this.keyboardToggle.bind(this);
    this.addKeyListener = this.addKeyListener.bind(this);
    this.removeKeyListener = this.removeKeyListener.bind(this);
  }

  componentDidMount() {
    if (stateManager) stateManager.connect(this);

    setInterval(this.update, 1000 / 60);
  }

  componentDidUpdate() {}
  update() {
    //draw time marker
  }

  keyboardToggle = () => {
    if (this.keyCheckRef.current.checked === true) {
      this.addKeyListener();
    } else {
      this.removeKeyListener();
    }
    this.setState({ keyboard_enabled: this.keyCheckRef.current.checked });
  };

  handleSubmit(event) {
    stateManager.setUrl(this.state.url);
    event.preventDefault();
  }

  logData(event) {
    stateManager.socket.emit("logData", this.state.data_logging_time);
    console.log("logData", this.state.data_logging_time);
  }

  stopRobot(event) {
    stateManager.socket.emit("stopRobot");
    console.log("Robot Stopped");
  }

  unstopRobot(event) {
    stateManager.socket.emit("unstopRobot");
    console.log("Robot UnStopped");
  }

  addKeyListener() {
    var map = {};
    this.onkey = function (e) {
      map[e.keyCode] = true;
    };
    document.addEventListener("keyup", this.onkey);
    let interval = 33.33;
    // keyHandler gets called every interval milliseconds
    this.intervalHandle = setInterval(stateManager.keyHandler, interval, map);
  }

  removeKeyListener() {
    document.removeEventListener("keyup", this.onkey);
    clearInterval(this.intervalHandle);
  }

  onDataLoggingTimeChange(value) {
    this.setState({ data_logging_time: value });
  }
  onYawChange(value) {
    this.setState({ yaw: value });
  }
  onMoveChange(value) {
    this.setState({ move: value });
  }

  onVelocityChange(value) {
    this.setState({ velocity: value });
  }

  handleClick(event) {
    const id = event.target.id;
    if (id === "move_joint_1") {
      stateManager.keyHandler({ 53: true });
    } else if (id === "move_joint_2") {
      stateManager.keyHandler({ 54: true });
    } else if (id === "move_joint_3") {
      stateManager.keyHandler({ 55: true });
    } else if (id === "move_joint_4") {
      stateManager.keyHandler({ 56: true });
    } else if (id === "move_joint_5") {
      stateManager.keyHandler({ 57: true });
    } else if (id === "move_joint_6") {
      stateManager.keyHandler({ 58: true });
    } else if (id === "go_home") {
      stateManager.keyHandler({ 59: true });
    } else if (id === "get_pos") {
      stateManager.keyHandler({ 60: true });
    } else if (id === "get_image") {
      stateManager.keyHandler({ 61: true });
    }
  }

  render() {
    return (
      <div ref={this.navRef}>
        <br />
        <br />
        <br />
        <div>
          <button id="move_joint_1" onClick={this.handleClick}>
            Move Joint 1
          </button>
          <button id="move_joint_2" onClick={this.handleClick}>
            Move Joint 2
          </button>
          <button id="move_joint_3" onClick={this.handleClick}>
            Move Joint 3
          </button>
        </div>
        <br />
        <div>
          <button id="move_joint_4" onClick={this.handleClick}>
            Move Joint 4
          </button>
          <button id="move_joint_5" onClick={this.handleClick}>
            Move Joint 5
          </button>
          <button id="move_joint_6" onClick={this.handleClick}>
            Move Joint 6
          </button>
        </div>
        <br />
        <br />

        <div>
          <button id="go_home" onClick={this.handleClick}>
            Go Home
          </button>
        </div>
        <br />
        <br />
        <div style={slider_style}>
          <label style={labelStyle}>Velocity: &nbsp;</label>
          <span>{this.state.velocity}</span>
          <br />
          <br />
          <Slider
            value={this.state.velocity}
            min={1}
            max={3}
            step={0.5}
            onChange={this.onVelocityChange}
          />
        </div>
        <br />
        <br />
        <button id="get_pos" onClick={this.handleClick}>
          {" "}
          Get End Effector Position
        </button>
        <br />
        <br />
        <label>
          {" "}
          X Pos <input type="text" value={this.state.ee_pos[0]} />{" "}
        </label>
        <br />
        <label>
          {" "}
          Y Pos <input type="text" value={this.state.ee_pos[1]} />{" "}
        </label>
        <br />
        <label>
          {" "}
          Z Pos <input type="text" value={this.state.ee_pos[2]} />{" "}
        </label>
        <br />
        <br />
        <Layer>
          <button id="get_image" onClick={this.handleClick}>
            GET IMAGE
          </button>
          <input type="text" value={this.state.velocity} />
        </Layer>
        <Rnd
          default={{
            x: 100,
            y: 300,
            width: 200,
            height: 200,
          }}
          lockAspectRatio={true}
        >
          <Stage width={256} height={256}>
            <Layer>
              <KImage image={this.state.image} width={256} height={256} />
            </Layer>
          </Stage>
        </Rnd>
      </div>
    );
  }
}

export default FrankaArmMover;
