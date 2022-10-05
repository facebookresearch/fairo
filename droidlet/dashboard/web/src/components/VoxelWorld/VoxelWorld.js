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
      flash_bbox: null,
    };
    this.worldContainerRef = React.createRef();
  }

  getVoxelWorldInitialState() {
    this.props.stateManager.worldSocket.emit("getVoxelWorldInitialState");
  }

  connectToWorld() {
    // this.props.stateManager.worldSocket.emit("connect");
    this.props.stateManager.worldSocket.emit("get_world_info"); // test world VW connection
  }

  flashVoxelWorldBlocks(bbox) {
    this.setState({ flash_bbox: bbox });
    this.worldContainerRef.current.contentWindow.postMessage(this.state, "*");
    this.setState({ flash_bbox: null });
  }

  componentDidUpdate() {
    this.worldContainerRef.current.contentWindow.postMessage(this.state, "*");
  }

  componentDidMount() {
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);

      this.worldContainerRef.current.contentWindow.addEventListener(
        "message",
        (event) => {
          const payload = event["data"];
          if (payload["status"] === "set_look") {
            this.props.stateManager.worldSocket.emit("set_look", payload);
          } else if (payload["status"] === "abs_move") {
            this.props.stateManager.worldSocket.emit("abs_move", payload);
          }
        },
        false
      );
    }

    this.getVoxelWorldInitialState();
    this.connectToWorld();

    // Listen for a message from the iframe to remove the prompt text when the user clicks in
    window.addEventListener(
      "message",
      (event) => {
        try {
          let data = JSON.parse(event.data);
          if (data.msg === "click") {
            let p = document.getElementById("prompt");
            if (p) {
              p.remove();
            }
          }
        } catch (e) {
          return false;
        }
      },
      false
    );
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
            height="1000"
            ref={this.worldContainerRef}
          ></iframe>
          <div id="prompt">
            <span id="prompt-text">Click in this window to enter 3D world</span>
            <br />
            <br />
            <span id="prompt-text">
              Control: press w/a/s/d to move, space to move up, shift to move
              down
            </span>
            <br />
            <br />
            <span id="prompt-text">Tip: press 'esc' to leave the 3D world</span>
          </div>
        </div>
        <p>
          Tip: press 'esc' to leave the 3D world, w/a/s/d to move, space to move
          up, shift to move down
        </p>
      </div>
    );
  }
}

export default VoxelWorld;
