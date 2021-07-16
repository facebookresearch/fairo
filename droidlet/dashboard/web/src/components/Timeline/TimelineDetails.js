/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { renderTable } from "./TimelineUtils";
import "./Timeline.css";

class TimelineDetails extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="subpanel">
        <table>
          <tbody>
            {renderTable(this.props.stateManager.memory.timelineDetails)}
          </tbody>
        </table>
      </div>
    );
  }
}

export default TimelineDetails;
