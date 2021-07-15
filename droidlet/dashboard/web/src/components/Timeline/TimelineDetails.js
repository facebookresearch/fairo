/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

class TimelineDetails extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div>
        <p>Placeholder for timeline details.</p>
      </div>
    );
  }
}

export default TimelineDetails;
