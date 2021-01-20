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
      <div id="world-container">
        <iframe
          id="ifr"
          src="VoxelWorld/world.html"
          title="Voxel World"
          width="1000"
          height="600"
          ref={this.worldContainerRef}
        ></iframe>
      </div>
    );
  }
}

export default VoxelWorld;
