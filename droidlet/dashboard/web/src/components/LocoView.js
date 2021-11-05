/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import LiveImage from "./LiveImage";
import LiveObjects from "./LiveObjects";
import LiveHumans from "./LiveHumans";

class LocoView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      rgb: null,
      depth: null,
      objects: null,
      humans: null,
    };
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  componentWillUnmount() {
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
  }

  render() {
    const stateManager = this.props.stateManager;

    return (
    <div>
        <LiveImage
        type={"rgb"}
        height={320}
        width={320}
        offsetH={10}
        offsetW={10}
        stateManager={stateManager}
        />
        <LiveImage
        type={"depth"}
        height={320}
        width={320}
        offsetH={10}
        offsetW={10 + 320 + 10}
        stateManager={stateManager}
        />
        <LiveObjects
        height={320}
        width={320}
        offsetH={10 + 320 + 10}
        offsetW={10}
        stateManager={stateManager}
        />
        <LiveHumans
        height={320}
        width={320}
        offsetH={10 + 320 + 10}
        offsetW={10 + 320 + 10}
        stateManager={stateManager}
        />
    </div>
    );
  }
}

export default LocoView;
