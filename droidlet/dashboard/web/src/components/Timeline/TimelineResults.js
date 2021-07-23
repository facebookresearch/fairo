/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { handleClick } from "./TimelineUtils";
import "./Timeline.css";

class TimelineResults extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  jsonToResultsTable(eventObj) {
    // which attribute to show as the description
    let descriptionTable = {
      perceive: "chat",
      interpreter: "tasks_to_push",
      dialogue: "object",
      memory: "operation",
    };

    let name = eventObj["name"];
    let time = eventObj["start_time"];
    let description = eventObj[descriptionTable[name]];
    // show the time and not the date
    time = time.substring(time.indexOf(" "));

    return (
      <table className="fixed">
        <tbody>
          <tr>
            <td>{time}</td>
            <td>{name}</td>
            <td>{description}</td>
          </tr>
        </tbody>
      </table>
    );
  }

  render() {
    return (
      <div className="subpanel">
        <table className="fixed">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Event Type</th>
              <th>Event Summary</th>
            </tr>
          </thead>
        </table>
        <hr />

        {this.props.stateManager.memory.timelineSearchResults &&
          this.props.stateManager.memory.timelineSearchResults.map(
            (item, index) => (
              <div
                className="result"
                key={index}
                onClick={() =>
                  handleClick(this.props.stateManager, JSON.stringify(item))
                }
              >
                {this.jsonToResultsTable(item)}
                <hr />
              </div>
            )
          )}
      </div>
    );
  }
}

export default TimelineResults;
