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
      yaw: 0.01, 
      velocity: 0.1,
    }

    this.handleSubmit = this.handleSubmit.bind(this);
    this.onYawChange = this.onYawChange.bind(this);
    this.onVelocityChange = this.onVelocityChange.bind(this);
    this.navRef = React.createRef();
    this.state = this.initialState
  }

  handleSubmit(event) {
    stateManager.setUrl(this.state.url);
    event.preventDefault();
  }

  componentDidMount() {
    if (stateManager) stateManager.connect(this);
    var map = {};
    var onkey = function (e) {
      map[e.keyCode] = true;
    };
    document.addEventListener("keyup", onkey);
    let interval = 33.33;
    setInterval(stateManager.keyHandler, interval, map); // keyHandler gets called every interval milliseconds
  }

  onYawChange(value) {
    this.setState({ yaw: value })
  }

  onVelocityChange(value) {
    this.setState({ velocity: value })
  }

  render() {
    return (
      <div ref={this.navRef}>
        <p> Use the arrow keys to move the robot around</p>
        <div style={slider_style}>
          <label style={labelStyle}>Yaw: &nbsp;</label>
          <span>{this.state.yaw}</span>
          <br />
          <br />
          <Slider
              value={this.state.yaw}
              min={0}
              max={0.5}
              step={0.01}
              onChange={this.onYawChange}
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
      </div>
    );
  }
}

export default Navigator;
