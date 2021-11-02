/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import stateManager from "../StateManager";
import "status-indicator/styles.css";
import Slider from "rc-slider";
import "rc-slider/assets/index.css";

let slider_style = { width: 600, margin: 50 };
const labelStyle = { minWidth: "60px", display: "inline-block" };

/**
 * Currently just a keybinding wrapper that hooks
 * into the state manager and sends events upstream.
 * Later, has to be expanded to visually acknowledge
 * the keys pressed, along with their successful receipt by
 * the backend.
 */
class Navigator extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      move: 0.3,
      yaw: 0.01,
      velocity: 0.1,
      data_logging_time: 30,
      keyboard_enabled: false,
    };

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

  keyboardToggle = () => {
    if (this.keyCheckRef.current.checked == true) {
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

  componentDidMount() {
    if (stateManager) stateManager.connect(this);
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
    if (id === "key_up") {
      stateManager.keyHandler({ 38: true });
    } else if (id === "key_down") {
      stateManager.keyHandler({ 40: true });
    } else if (id === "key_left") {
      stateManager.keyHandler({ 37: true });
    } else if (id === "key_right") {
      stateManager.keyHandler({ 39: true });
    } else if (id === "pan_left") {
      stateManager.keyHandler({ 49: true });
    } else if (id === "pan_right") {
      stateManager.keyHandler({ 50: true });
    } else if (id === "tilt_up") {
      stateManager.keyHandler({ 51: true });
    } else if (id === "tilt_down") {
      stateManager.keyHandler({ 52: true });
    }
  }

  // <button
  //   id="stop_robot"
  //   style={{ fontSize: 48 + "px" }}
  //   onClick={this.stopRobot}
  // >
  //   <strike>STOP ROBOT</strike>
  // </button>
  // <button id="unstop_robot" onClick={this.unstopRobot}>
  //   Clear Runstop
  // </button>

  render() {
    return (
      <div ref={this.navRef}>
        <br />
        <br />
        <div>
          <label> Base </label>
          <button id="key_up" onClick={this.handleClick}>
            Up
          </button>
        </div>
        <div>
          <button id="key_left" onClick={this.handleClick}>
            Left
          </button>
          <button id="key_down" onClick={this.handleClick}>
            Down
          </button>
          <button id="key_right" onClick={this.handleClick}>
            Right
          </button>
        </div>
        <br />
        <br />
        <div>
          <label> Camera </label>
          <button id="tilt_up" onClick={this.handleClick}>
            Up
          </button>
        </div>
        <div>
          <button id="pan_left" onClick={this.handleClick}>
            Left
          </button>
          <button id="tilt_down" onClick={this.handleClick}>
            Down
          </button>
          <button id="pan_right" onClick={this.handleClick}>
            Right
          </button>
        </div>

        <br />
        <input
          type="checkbox"
          defaultChecked={this.state.keyboard_enabled}
          ref={this.keyCheckRef}
          onChange={this.keyboardToggle}
        />
        <label>
          {" "}
          Enable Keyboard control (Use the arrow keys to move the robot, keys{" "}
          {(1, 2)} and {(3, 4)} to move camera){" "}
        </label>
        <div style={slider_style}>
          <label style={labelStyle}>Rotation (radians): &nbsp;</label>
          <span>{this.state.yaw}</span>
          <br />
          <br />
          <Slider
            value={this.state.yaw}
            min={0}
            max={6.28}
            step={0.01}
            onChange={this.onYawChange}
          />
          <br />
          <label style={labelStyle}>Move Distance (metres): &nbsp;</label>
          <span>{this.state.move}</span>
          <br />
          <br />
          <Slider
            value={this.state.move}
            min={0}
            max={10}
            step={0.1}
            onChange={this.onMoveChange}
          />
        </div>
        <div style={slider_style}>
          <label style={labelStyle}>Velocity: &nbsp;</label>
          <span>{this.state.velocity}</span>
          <br />
          <br />
          <Slider
            value={this.state.velocity}
            min={0}
            max={1}
            step={0.05}
            onChange={this.onVelocityChange}
          />
        </div>
        <div>
          <div style={slider_style}>
            <label style={labelStyle}>
              Data Logging Time (seconds): &nbsp;
            </label>
            <span>{this.state.data_logging_time}</span>
            <br />
            <br />
            <Slider
              value={this.state.data_logging_time}
              min={0}
              max={300}
              step={1}
              onChange={this.onDataLoggingTimeChange}
            />
          </div>
          <button id="log_data" onClick={this.logData}>
            Log Data
          </button>
        </div>
      </div>
    );
  }
}

export default Navigator;
