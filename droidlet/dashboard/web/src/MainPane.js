/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import InteractApp from "./components/Interact/InteractApp";
import VoxelWorld from "./components/VoxelWorld/VoxelWorld";

class MainPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      rgb: null,
      depth: null,
      objects: null,
      humans: null,
    };
  }

  render() {
    const stateManager = this.props.stateManager;

    return (
      <div>
        <InteractApp stateManager={stateManager} />
        <VoxelWorld stateManager={stateManager} />
      </div>
    );
  }
}

export default MainPane;
