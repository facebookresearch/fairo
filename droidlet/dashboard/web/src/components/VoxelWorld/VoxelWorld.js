/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "./VoxelWorld.css";

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

    // Listen for a message from the iframe to remove the prompt text when the user clicks in
    window.addEventListener("message", (event) => {
      try {
        let data = JSON.parse(event.data);
        if (data.msg === "click") {
          let p = document.getElementById("prompt");
          if (p) {p.remove()};
        }
      } catch (e) {
          return false
      }
    }, false);
  }

  render() {
    return (
      <div>
        <div id="world-container">
          <iframe
            id="ifr"
            src="VoxelWorld/world.html"
            title="Voxel World"
            width="100%"
            height="500"
            ref={this.worldContainerRef}
          ></iframe>
          <div id="prompt">
            <span id="prompt-text">Click in this window to enter 3D world</span>
          </div>
        </div>
        <p>
          Tip: press 'esc' to leave the 3D world, w/a/s/d to move, space to
          jump
        </p>
      </div>
    );
  }
}

export default VoxelWorld;
