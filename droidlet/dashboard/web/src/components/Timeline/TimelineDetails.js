/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { renderTable } from "./TimelineUtils";
import "./Timeline.css";

class TimelineDetails extends React.Component {
  constructor(props) {
    super(props);
    this.update = false;
  }

  shouldComponentUpdate() {
    // flag to prevent rerendering
    return !this.update;
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    this.update = true;
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
