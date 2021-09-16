/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import LiveImage from "./components/LiveImage";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
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
      agentType: 'locobot',
    };
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    console.log("MainPane rendering agent type: " + this.state.agentType);
    const stateManager = this.props.stateManager;

    if (this.state.agentType === 'locobot'){
      return (
        <div>
          <InteractApp stateManager={stateManager} />
          <LiveImage
            type={"rgb"}
            height={320}
            width={320}
            offsetH={320 + 80}
            offsetW={10}
            stateManager={stateManager}
          />
          <LiveImage
            type={"depth"}
            height={320}
            width={320}
            offsetH={320 + 80}
            offsetW={10 + 320 + 10}
            stateManager={stateManager}
          />
          <LiveObjects
            height={320}
            width={320}
            offsetH={320 + 60 + 320 + 30}
            offsetW={10}
            stateManager={stateManager}
          />
          <LiveHumans
            height={320}
            width={320}
            offsetH={320 + 60 + 320 + 30}
            offsetW={10 + 320 + 10}
            stateManager={stateManager}
          />
        </div>
      );
    }
    else if (this.state.agentType === 'craftassist'){
      return (
        <div>
          <InteractApp stateManager={stateManager} />
          <VoxelWorld
            stateManager={stateManager}
          />
        </div>
      );
    }
    else {
      console.log("MainPane received invalid agent type");
      return (
        <h1>Error: Invalid agent type</h1>
      );
    }
  }
}

export default MainPane;
