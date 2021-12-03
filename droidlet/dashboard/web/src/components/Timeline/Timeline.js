/*
Copyright (c) Facebook, Inc. and its affiliates.

Timeline.js displays a timeline of agent activities on the dashboard, 
relying on the visjs Timeline framework. Enable this when running the 
agent using the flags --enable_timeline --log_timeline.
*/

import React, { createRef } from "react";
import { Timeline, DataSet } from "vis-timeline/standalone";
import { handleClick, capitalizeEvent } from "./TimelineUtils";
import { handleSearch } from "./TimelineSearch";
import TimelineDropdown from "./TimelineDropdown";
import SearchIcon from "@material-ui/icons/Search";
import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
import "./Timeline.css";

const theme = createMuiTheme({
  palette: {
    type: "dark",
  },
});

const timelineEvents = new DataSet();

const timelineTypes = ["Perceive", "Dialogue", "Interpreter", "Memory"];

const groups = [
  {
    id: "Timeline",
    content: "Timeline",
    nestedGroups: timelineTypes,
  },
  {
    id: "Perceive",
    content: "Perception",
  },
  {
    id: "Dialogue",
    content: "Dialogue",
  },
  {
    id: "Interpreter",
    content: "Interpreter",
  },
  {
    id: "Memory",
    content: "Memory",
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
    this.searchPattern = "";
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);

    // make a shallow copy of search filters
    this.searchFilters = [...this.props.stateManager.memory.timelineFilters];

    this.timeline = new Timeline(
      this.appRef.current,
      timelineEvents,
      groups,
      options
    );
    // set current viewing window to 20 seconds for readability
    let startTime = this.timeline.getCurrentTime();
    let endTime = this.timeline.getCurrentTime();
    startTime = startTime.setSeconds(startTime.getSeconds() - 10);
    endTime = endTime.setSeconds(endTime.getSeconds() + 10);
    this.timeline.setOptions({
      start: startTime,
      end: endTime,
    });

    // store this keyword to access it inside the event handler
    const that = this;
    this.timeline.on("click", function (properties) {
      if (properties["item"]) {
        const item = timelineEvents.get(properties["item"]);
        handleClick(that.props.stateManager, item.title);
      }
    });
  }

  renderEvent() {
    const event = this.props.stateManager.memory.timelineEvent;
    // prevents duplicates because state changes cause the page to rerender
    if (event && event !== this.prevEvent) {
      this.prevEvent = event;
      const eventObj = JSON.parse(event);
      let description = "";
      if (eventObj["name"] === "perceive") {
        description = ' ("' + eventObj["chat"] + '")';
      }

      // adds to the outer timeline group
      timelineEvents.add([
        {
          title: JSON.stringify(eventObj, null, 2),
          content: eventObj["name"] + description,
          group: "Timeline",
          className: eventObj["name"],
          start: eventObj["start_time"],
          end: eventObj["end_time"],
          type: "box",
        },
      ]);
      // adds the same item to the inner nested group
      timelineEvents.add([
        {
          title: JSON.stringify(eventObj, null, 2),
          group: capitalizeEvent(eventObj["name"]),
          className: eventObj["name"],
          start: eventObj["start_time"],
          end: eventObj["end_time"],
          type: "box",
        },
      ]);
    }
  }

  toggleVisibility() {
    const filters = this.props.stateManager.memory.timelineFilters;
    // checks if filters have been changed
    if (filters && filters !== this.searchFilters) {
      //console.log(filters);
      this.searchFilters = [...filters];
      let items = timelineEvents.get();
      // loop through all items and check if the filter applies
      for (let i = 0; i < items.length; i++) {
        if (filters.includes(capitalizeEvent(items[i].className))) {
          items[i].style = "opacity: 1;";
        } else {
          items[i].style = "opacity: 0.2;";
        }
      }
      timelineEvents.update(items);
    }
  }

  render() {
    this.renderEvent();
    this.toggleVisibility();
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
          onChange={(e) =>
            handleSearch(this.props.stateManager, e.target.value)
          }
        />

        <div id="dropdown">
          <ThemeProvider theme={theme}>
            <TimelineDropdown stateManager={this.props.stateManager} />
          </ThemeProvider>
        </div>

        <div className="clearfloat"></div>

        <div ref={this.appRef} />
      </div>
    );
  }
}

export default DashboardTimeline;
export { timelineTypes };
