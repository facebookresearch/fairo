import React from "react";

class MobileDirectionButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      commands: [],
    };
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
