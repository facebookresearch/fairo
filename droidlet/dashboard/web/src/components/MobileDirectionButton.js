import React from "react";
import stateManager from "../StateManager";

class MobileDirectionButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      commands: [],
    };
  }

  componentDidMount() {
    // constantly tells stateManager to handle button presses
    // logic is similar to that in ./Navigator.js
    // need to bind this so this.state within sendAndClearCommands refers to the correct object
    setInterval(this.sendAndClearCommands.bind(this), 33.33);
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
      <div>
        <button onClick={() => this.addCommand("MOVE_LEFT")}>LEFT</button>
        <button onClick={() => this.addCommand("MOVE_FORWARD")}>UP </button>
        <button onClick={() => this.addCommand("MOVE_DOWN")}>DOWN </button>
        <button onClick={() => this.addCommand("MOVE_RIGHT")}>RIGHT</button>
      </div>
    );
  }
}

export default MobileDirectionButton;
