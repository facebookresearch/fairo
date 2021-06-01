import React from "react";

class MobileDirectionButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      commands: [],
    };
  }

  addCommand(command) {
    console.log("added command");
    let prevCommand = this.state.commands;
    prevCommand.push(command);
    this.setState({
      commands: prevCommand,
    });
    console.log("new state is");
    console.log(this.state);
  }

  render() {
    return (
      <div>
        <button onClick={() => this.addCommand("MOVE_LEFT")} class="button">
          LEFT
        </button>
        <button onClick={() => this.addCommand("MOVE_FORWARD")} class="button">
          UP{" "}
        </button>
        <button onClick={() => this.addCommand("MOVE_DOWN")} class="button">
          DOWN{" "}
        </button>
        <button onClick={() => this.addCommand("MOVE_RIGHT")} class="button">
          RIGHT
        </button>
      </div>
    );
  }
}

export default MobileDirectionButton;
