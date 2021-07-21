/*
Copyright (c) Facebook, Inc. and its affiliates.

Timeline.js displays a timeline of agent activities on the dashboard, 
relying on the visjs Timeline framework. Enable this when running the 
agent using the flags --enable_timeline --log_timeline.
*/

import React, { createRef } from "react";
import Fuse from "fuse.js";
import { Timeline, DataSet } from "vis-timeline/standalone";
import { jsonToArray, capitalizeEvent } from "./TimelineUtils";
import SearchIcon from "@material-ui/icons/Search";
import "./Timeline.css";

const items = new DataSet();

const groups = [
  {
    id: "timeline",
    content: "Timeline",
    nestedGroups: ["perceive", "dialogue", "interpreter"],
  },
  {
    id: "perceive",
    content: "Perception",
  },
  {
    id: "dialogue",
    content: "Dialogue",
  },
  {
    id: "interpreter",
    content: "Interpreter",
  },
];

const options = {
  tooltip: {
    followMouse: true,
    overflowMethod: "cap",
    template: function (originalItemData, parsedItemData) {
      const titleJSON = JSON.parse(originalItemData.title);
      return (
        "<pre>event: " +
        titleJSON.name +
        "\nagent time: " +
        titleJSON.agent_time +
        "</pre>"
      );
    },
  },
  zoomMax: 86400000,
  rollingMode: {
    follow: true,
  },
  stack: false,
};

const SearchBar = ({ onChange, placeholder }) => {
  return (
    <div className="search">
      <input
        className="searchInput"
        type="text"
        onChange={onChange}
        placeholder={placeholder}
      />
      <span className="searchSpan">
        <SearchIcon />
      </span>
    </div>
  );
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
    this.timeline = new Timeline(this.appRef.current, items, groups, options);
    // set current viewing window to 10 seconds for readability
    let currentTime = this.timeline.getCurrentTime();
    this.timeline.setOptions({
      start: currentTime.setSeconds(currentTime.getSeconds() - 5),
      end: currentTime.setSeconds(currentTime.getSeconds() + 10),
    });
    // store this keyword to access it inside the event handler
    const that = this;
    this.timeline.on("click", function (properties) {
      if (properties["item"]) {
        const item = items.get(properties["item"]);
        that.handleClick(item);
      }
    });
  }

  // TODO: combine this handleClick() with the one in TimelineUtils.js
  handleClick(item) {
    const eventObj = JSON.parse(item.title);
    let tableArr = jsonToArray(eventObj);
    this.props.stateManager.memory.timelineDetails = tableArr;
    this.props.stateManager.updateTimeline();

    var config = {
      title: capitalizeEvent(eventObj["name"]),
      cssClass: "scrollable",
      type: "react-component",
      component: "TimelineDetails",
      props: { stateManager: this.props.stateManager },
    };
    this.props.stateManager.dashboardLayout.root.contentItems[0].contentItems[1].contentItems[0].contentItems[5].contentItems[1].contentItems[1].addChild(
      config
    );
  }

  handleSearch(pattern) {
    const matches = [];
    if (pattern) {
      const fuseOptions = {
        // set ignoreLocation to true or else it searches the first 60 characters by default
        ignoreLocation: true,
        useExtendedSearch: true,
      };

      const fuse = new Fuse(
        this.props.stateManager.memory.timelineEventHistory,
        fuseOptions
      );

      // prepending Fuse operator to search for results that include the pattern
      const result = fuse.search("'" + pattern);

      if (result.length) {
        result.forEach(({ item }) => {
          const eventObj = JSON.parse(item);
          matches.push(eventObj);
        });
      }
    }
    this.props.stateManager.memory.timelineSearchResults = matches;
    this.props.stateManager.updateTimeline();
  }

  renderEvent() {
    const event = this.props.stateManager.memory.timelineEvent;
    // prevents duplicates because state changes cause the page to rerender
    if (event && event !== this.prevEvent) {
      this.prevEvent = event;
      const eventObj = JSON.parse(event);
      // adds to the outer timeline group
      items.add([
        {
          title: JSON.stringify(eventObj, null, 2),
          content: eventObj["name"],
          group: "timeline",
          className: eventObj["name"],
          start: eventObj["start_time"],
          end: eventObj["end_time"],
        },
      ]);
      // adds the same item to the inner nested group
      items.add([
        {
          title: JSON.stringify(eventObj, null, 2),
          group: eventObj["name"],
          className: eventObj["name"],
          start: eventObj["start_time"],
          end: eventObj["end_time"],
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
          <br />
          Click an event to view more details!
        </p>

        <SearchBar
          placeholder="Search"
          onChange={(e) => this.handleSearch(e.target.value)}
        />

        <div ref={this.appRef} />
      </div>
    );
  }
}

export default DashboardTimeline;
