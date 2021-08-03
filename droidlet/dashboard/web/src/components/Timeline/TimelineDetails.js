/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import "./Timeline.css";

class TimelineDetails extends React.Component {
  shouldComponentUpdate() {
    // prevent rerendering after the initial click event
    return false;
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  renderTable(tableArr) {
    // returns an HTML table given an array
    if (tableArr) {
      return tableArr.map((data, index) => {
        const { event, description } = data;
        return (
          <tr key={index}>
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
    return (
      <div className="subpanel">
        <table>
          <tbody>
            {this.renderTable(this.props.stateManager.memory.timelineDetails)}
          </tbody>
        </table>
      </div>
    );
  }
}

export default TimelineDetails;
