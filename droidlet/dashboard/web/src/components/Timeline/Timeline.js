/*
Copyright (c) Facebook, Inc. and its affiliates.

Timeline.js displays a timeline of agent activities on the dashboard, 
relying on the visjs Timeline framework. Enable this when running the 
agent using the flags --enable_timeline --log_timeline.
*/

import React, { createRef } from "react";
import Fuse from "fuse.js";
import { Timeline, DataSet } from "vis-timeline/standalone";
import SearchIcon from "@material-ui/icons/Search";
import "vis-timeline/styles/vis-timeline-graph2d.css";
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
    this.state = {
      // used to construct table on click
      tableBody: [],
      // used to return results from search
      searchResults: [],
    };
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

  handleClick(item) {
    const eventObj = JSON.parse(item.title);
    let tableArr = this.jsonToArray(eventObj);
    this.setState({
      searchResults: [],
      tableBody: tableArr,
    });
  }

  jsonToArray(eventObj) {
    // turns JSON hook data into an array that can easily be turned into an HTML table
    let tableArr = [];
    for (let key in eventObj) {
      if (eventObj.hasOwnProperty(key)) {
        // stringify JSON object for logical form
        if (key === "logical_form") {
          tableArr.push({
            event: this.capitalizeEvent(key),
            description: JSON.stringify(eventObj[key]),
          });
        } else {
          tableArr.push({
            event: this.capitalizeEvent(key),
            description: eventObj[key],
          });
        }
      }
    }
    return tableArr;
  }

  handleSearch(pattern) {
    if (pattern) {
      const fuse = new Fuse(
        this.props.stateManager.memory.timelineEventHistory
      );
      const result = fuse.search(pattern);
      const matches = [];
      if (!result.length) {
        // empty results pane
        this.setState({
          tableBody: [],
          searchResults: [],
        });
      } else {
        result.forEach(({ item }) => {
          const eventObj = JSON.parse(item);
          matches.push(eventObj);
        });
        // set pane to show matches
        this.setState({
          tableBody: [],
          searchResults: matches,
        });
      }
    } else {
      // empty results pane
      this.setState({
        tableBody: [],
        searchResults: [],
      });
    }
  }

  capitalizeEvent(str) {
    // replaces underscores with spaces
    str = str.replace(/_/g, " ");
    // capitalizes the first letter of every word
    return str.replace(/\w\S*/g, function (txt) {
      return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    });
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

  renderClickTable() {
    return this.renderTable(this.state.tableBody);
  }

  renderTable(tableArr) {
    if (tableArr) {
      return tableArr.map((data) => {
        const { event, description } = data;
        return (
          <tr>
            <td>
              <strong>{event}</strong>
            </td>
            <td>{description}</td>
          </tr>
        );
      });
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

        <SearchBar
          placeholder="Search"
          onChange={(e) => this.handleSearch(e.target.value)}
        />

        <div ref={this.appRef} />

        <div className="item">
          <p id="result">Results:</p>
          <table>
            <tbody>{this.renderClickTable()}</tbody>
          </table>

          <div className="matches">
            {this.state.searchResults.map((item) => (
              <div>
                {this.renderTable(this.jsonToArray(item))}
                <hr />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
}

export default DashboardTimeline;
