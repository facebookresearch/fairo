/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { renderTable, jsonToArray, handleClick } from "./TimelineUtils";
import "./Timeline.css";

class TimelineResults extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="subpanel">
        {this.props.stateManager.memory.timelineSearchResults.map((item) => (
          <div
            className="result"
            onClick={() =>
              handleClick(this.props.stateManager, JSON.stringify(item))
            }
          >
            {renderTable(jsonToArray(item))}
            <hr />
          </div>
        ))}
      </div>
    );
  }
}

export default TimelineResults;
