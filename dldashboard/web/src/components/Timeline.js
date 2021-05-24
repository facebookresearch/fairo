/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

class Timeline extends React.Component {
  renderHandshake() {
    this.props.stateManager.socket.emit("receiveHandshake", "Sent message!");
    return this.props.stateManager.memory.handshake;
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="timeline">
        <p>
          An agent activity visualizer where users can easily view, inspect and
          search through agent activities interactively.
        </p>
        <p>Handshake status: {this.renderHandshake()}</p>
      </div>
    );
  }
}

export default Timeline;
