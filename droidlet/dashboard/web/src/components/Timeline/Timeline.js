/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

class Timeline extends React.Component {
  renderHandshake() {
    this.props.stateManager.socket.emit(
      "receiveTimelineHandshake",
      "Sent message!"
    );
    return this.props.stateManager.memory.timelineHandshake;
  }

  renderEvents() {
    return this.props.stateManager.memory.timelineEvents;
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="timeline">
        <p>
          A visualizer where users can easily view, inspect, and search through
          agent activities interactively.
        </p>
        <p>Handshake status: {this.renderHandshake()}</p>
        <p>Memory event: {this.renderEvents()}</p>
      </div>
    );
  }
}

export default Timeline;
