/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "./History.css";

class Timeline extends React.Component {
  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="timeline">
        <p>
          An agent activity visualizer where the users can easily view, inspect
          and search through agent activities interactively.
        </p>
      </div>
    );
  }
}

export default Timeline;
