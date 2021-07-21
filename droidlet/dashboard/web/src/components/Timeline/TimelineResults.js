/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { handleClick, jsonToResultsTable } from "./TimelineUtils";
import "./Timeline.css";

class TimelineResults extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
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

        {this.props.stateManager.memory.timelineSearchResults.map(
          (item, index) => (
            <div
              className="result"
              key={index}
              onClick={() =>
                handleClick(this.props.stateManager, JSON.stringify(item))
              }
            >
              {jsonToResultsTable(item)}
              <hr />
            </div>
          )
        )}
      </div>
    );
  }
}

export default TimelineResults;
