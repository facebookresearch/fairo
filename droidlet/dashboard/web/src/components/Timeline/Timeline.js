/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React, { createRef } from "react";
import { Timeline, DataSet } from "vis-timeline/standalone";
import "vis-timeline/styles/vis-timeline-graph2d.css";
import "./Timeline.css";

// const items = new DataSet([
//   { content: "item 1", start: "2021-06-21 14:50:10.802386", type: "point" },
//   { content: "item 2", start: "2021-06-21 14:50:10", type: "point" },
// ]);

const items = new DataSet();

const options = {
  // start: "2021-06-21 14:46:00",
  // end: "2021-06-21 16:46:00",
};

class DashboardTimeline extends React.Component {
  constructor() {
    super();
    this.timeline = {};
    this.appRef = createRef();
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    this.timeline = new Timeline(this.appRef.current, items, options);
    this.renderEventHistory();
  }

  // componentDidUpdate() {
  //   this.renderEventHistory();
  // }

  renderHandshake() {
    this.props.stateManager.socket.emit(
      "receiveTimelineHandshake",
      "Sent message!"
    );
    return this.props.stateManager.memory.timelineHandshake;
  }

  renderEvent() {
    const event = this.props.stateManager.memory.timelineEvent;
    if (event) {
      const eventObj = JSON.parse(event);
      if (eventObj["name"] === "perceive") {
        items.add([
          { content: event, start: eventObj["datetime"], type: "point" },
        ]);
      }
    }
    return event;
  }

  renderEventHistory() {
    // doesn't work, as in nothing from history gets rendered
    const eventHistory = this.props.stateManager.memory.timelineEventHistory;
    for (let i = 0; i < eventHistory.length; i++) {
      const event = eventHistory[i];
      const eventObj = JSON.parse(event);
      if (eventObj["name"] === "perceive") {
        items.add([
          { content: event, start: eventObj["datetime"], type: "point" },
        ]);
      }
    }
    return eventHistory;
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
        {/* <p>Memory history: </p> */}
        <ul>
          {/* {this.renderEventHistory().map((item) => (
            <li>{item}</li>
          ))} */}
        </ul>
      </div>
    );
  }
}

export default DashboardTimeline;
