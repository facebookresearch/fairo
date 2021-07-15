/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

class TimelineResults extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div>
        <p>Placeholder for timeline results.</p>
      </div>
    );
  }
}

export default TimelineResults;
