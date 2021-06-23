/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React, { createRef } from "react";
import { Timeline, DataSet } from "vis-timeline/standalone";
import "vis-timeline/styles/vis-timeline-graph2d.css";
import "./Timeline.css";

const items = new DataSet();

const options = {};

class DashboardTimeline extends React.Component {
  constructor() {
    super();
    this.timeline = {};
    this.appRef = createRef();
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    this.timeline = new Timeline(this.appRef.current, items, options);
  }

  renderHandshake() {
    this.props.stateManager.socket.emit(
      "receiveTimelineHandshake",
      "Sent message!"
    );
    return this.props.stateManager.memory.timelineHandshake;
  }

  renderEvent() {
    this.addEvent();
    return this.props.stateManager.memory.timelineEvent;
  }

  addEvent() {
    const event = this.props.stateManager.memory.timelineEvent;
    if (event) {
      const eventObj = JSON.parse(event);
      if (
        items.length <
        this.props.stateManager.memory.timelineEventHistory.length
      ) {
        items.add([
          {
            content: eventObj["name"],
            start: eventObj["datetime"],
          },
        ]);
      }
    }
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
        <div ref={this.appRef} />
      </div>
    );
  }
}

export default DashboardTimeline;
