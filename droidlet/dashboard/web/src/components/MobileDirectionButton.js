/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import stateManager from "../StateManager";
import "./MobileDirectionButton.css";

class MobileDirectionButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      commands: [],
    };
    this.intervalId = undefined;
  }

  componentDidMount() {
    // constantly tells stateManager to handle button presses
    // logic is similar to that in ./Navigator.js
    // need to bind this so this.state within sendAndClearCommands refers to the correct this object
    let interval = 33.33;
    this.intervalId = setInterval(
      this.sendAndClearCommands.bind(this),
      interval
    ); // sendAndClearCommmands gets called every interval milliseconds
  }

  componentWillUnmount() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }

  /**
   * This function is called when a button is pressed.
   * Adds the corresponding button's command to the list of commands to be executed by the backend
   */
  addCommand(command) {
    let prevCommand = this.state.commands;
    prevCommand.push(command);
    this.setState({
      commands: prevCommand,
    });
  }

  /**
   * sends commands to stateManager
   * clears commands
   */
  sendAndClearCommands() {
    if (this.state) {
      stateManager.buttonHandler(this.state.commands);
      // clears the commands once sent
      this.setState({
        commands: [],
      });
    }
  }

  render() {
    return (
      <div className="container">
        <button
          className="directionButton left"
          onClick={() => this.addCommand("MOVE_LEFT")}
        >
          Turn Left
        </button>
        <button
          className="directionButton up"
          onClick={() => this.addCommand("MOVE_FORWARD")}
        >
          Forward
        </button>
        <button
          className="directionButton down"
          onClick={() => this.addCommand("MOVE_DOWN")}
        >
          Backward
        </button>
        <button
          className="directionButton right"
          onClick={() => this.addCommand("MOVE_RIGHT")}
        >
          Turn Right
        </button>
      </div>
    );
  }
}

export default MobileDirectionButton;
