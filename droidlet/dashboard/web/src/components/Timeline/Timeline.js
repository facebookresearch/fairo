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

  renderEvent() {
    return this.props.stateManager.memory.timelineEvent;
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
        <p>Latest memory event: {this.renderEvent()}</p>
      </div>
    );
  }
}

export default Timeline;
