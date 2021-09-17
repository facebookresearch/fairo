/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import InteractApp from "./components/Interact/InteractApp";
import LocoView from "./components/LocoView";
import VoxelWorld from "./components/VoxelWorld/VoxelWorld";

class MainPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
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
          <LocoView stateManager={stateManager} />
        </div>
      );
    }
    else if (this.state.agentType === 'craftassist'){
      return (
        <div>
          <InteractApp stateManager={stateManager} />
          <VoxelWorld stateManager={stateManager} />
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
