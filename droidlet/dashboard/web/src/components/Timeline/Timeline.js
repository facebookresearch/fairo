/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React, { createRef } from "react";
import { Timeline, DataSet } from "vis-timeline/standalone";
import "./Timeline.css";

const items = new DataSet([
  { content: "item 1", start: "2021-06-18 17:41:00.555555", type: "point" },
  { content: "item 2", start: "2021-06-18 17:42:00", type: "point" },
  { content: "item 3", start: "2021-06-18 17:43:00", type: "point" },
  { content: "item 4", start: "2021-06-18 17:44:00", type: "point" },
  { content: "item 5", start: "2021-06-18 17:45:00", type: "point" },
  { content: "item 6", start: "2021-06-18 17:46:00", type: "point" },
]);

// const items = new DataSet();

const options = {
  timeAxis: { scale: "minute", step: 1 },
};

class DashboardTimeline extends React.Component {
  constructor() {
    super();
    this.timeline = {};
    this.eventHistory = [];
    this.appRef = createRef();
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    this.timeline = new Timeline(this.appRef.current, items, options);
  }

  componentDidUpdate() {}

  renderHandshake() {
    this.props.stateManager.socket.emit(
      "receiveTimelineHandshake",
      "Sent message!"
    );
    return this.props.stateManager.memory.timelineHandshake;
  }

  renderEvent() {
    const event = this.props.stateManager.memory.timelineEvent;
    // if (event)
    // {
    //   const eventObj = JSON.parse(event);
    //   if (eventObj["name"] === "perceive")
    //   {
    //     items.add([
    //       {content: event, start: eventObj["datetime"], type: "point" }
    //     ])
    //   }
    // }
    return event;
  }

  renderEventHistory() {
    this.eventHistory = this.props.stateManager.memory.timelineEventHistory;
    // for (let i = 0; i < this.eventHistory.length; i++)
    // {
    //   const event = this.eventHistory[i];
    //   const eventObj = JSON.parse(event);
    //   if (eventObj["name"] === "perceive")
    //   {
    //     items.add([
    //       { content: event, start: eventObj["datetime"], type: "point" }
    //     ])
    //   }
    // }
    return this.eventHistory;
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
        <p>Memory history: </p>
        <ul>
          {this.renderEventHistory().map((item) => (
            <li>{item}</li>
          ))}
        </ul>
      </div>
    );
  }
}

export default DashboardTimeline;
