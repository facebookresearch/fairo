/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { renderTable, jsonToArray } from "./TimelineUtils";
import "./Timeline.css";

class TimelineResults extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="subpanel">
        {this.props.stateManager.memory.timelineSearchResults.map((item) => (
          <div>
            {renderTable(jsonToArray(item))}
            <hr />
          </div>
        ))}
      </div>
    );
  }
}

export default TimelineResults;
