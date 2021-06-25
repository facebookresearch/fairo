/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React, { createRef } from "react";
import { Timeline, DataSet } from "vis-timeline/standalone";
import "vis-timeline/styles/vis-timeline-graph2d.css";
import "./Timeline.css";

const items = new DataSet();

const options = {
  tooltip: {
    followMouse: true,
    overflowMethod: "cap",
    // preserves the formatting from JSON.stringify()
    template: function (originalItemData, parsedItemData) {
      return "<pre>" + originalItemData.title + "</pre>";
    },
  },
};

class DashboardTimeline extends React.Component {
  constructor() {
    super();
    this.timeline = {};
    this.appRef = createRef();
    this.prevEvent = "";
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    this.timeline = new Timeline(this.appRef.current, items, options);
  }

  renderEvent() {
    const event = this.props.stateManager.memory.timelineEvent;
    // prevents duplicates because state changes cause the page to rerender
    if (event && event !== this.prevEvent) {
      this.prevEvent = event;
      const eventObj = JSON.parse(event);
      items.add([
        {
          title: JSON.stringify(eventObj, null, 2),
          content: eventObj["name"],
          start: eventObj["datetime"],
          selectable: false,
        },
      ]);
    }
  }

  render() {
    this.renderEvent();
    return (
      <div className="timeline">
        <p id="description">
          A visualizer for viewing, inspecting, and searching through agent
          activities interactively.
        </p>
        <div ref={this.appRef} />
      </div>
    );
  }
}

export default DashboardTimeline;
