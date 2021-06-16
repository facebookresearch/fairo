/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

class VoxelWorld extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      status: "",
      world_state: {},
    };
    this.worldContainerRef = React.createRef();
  }

  getVoxelWorldInitialState() {
    this.props.stateManager.socket.emit("getVoxelWorldInitialState");
  }

  componentDidUpdate() {
    this.worldContainerRef.current.contentWindow.postMessage(this.state, "*");
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
    this.getVoxelWorldInitialState();
  }

  render() {
    return (
      <div>
        <div id="world-container">
          <iframe
            id="ifr"
            src="VoxelWorld/world.html"
            title="Voxel World"
            width="900"
            height="500"
            ref={this.worldContainerRef}
          ></iframe>
        </div>
        <p>
          Tip: press 'esc' to leave the voxel world, w/a/s/d to move, space to
          jump
        </p>
      </div>
    );
  }
}

export default VoxelWorld;
