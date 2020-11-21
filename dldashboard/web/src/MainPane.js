/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import LiveImage from "./components/LiveImage";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import InteractApp from "./components/Interact/InteractApp";

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
        <LiveImage
          height={320}
          width={320}
          offsetH={320 + 30}
          offsetW={10}
          stateManager={stateManager}
        />
        <LiveImage
          type={"depth"}
          height={320}
          width={320}
          offsetH={320 + 30}
          offsetW={10 + 320 + 10}
          stateManager={stateManager}
        />
        <LiveObjects
          height={320}
          width={320}
          offsetH={320 + 30 + 320 + 30}
          offsetW={10}
          stateManager={stateManager}
        />
        <LiveHumans
          height={320}
          width={320}
          offsetH={320 + 30 + 320 + 30}
          offsetW={10 + 320 + 10}
          stateManager={stateManager}
        />
      </div>
    );
  }
}

export default MainPane;
