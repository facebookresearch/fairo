/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/World3D.js

import React from "react";
import Worldview, { Cubes, Axes } from "regl-worldview";

var hashCode = function (s) {
  return s.split("").reduce(function (a, b) {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);
};

class World3D extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      height: 400,
      width: 600,
      isLoaded: false,
    };
    this.state = this.initialState;
    this.outer_div = React.createRef();
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }
  componentWillUnmount() {
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
  }

  render() {
    const markers = [
      {
        pose: {
          orientation: { x: 0, y: 0, z: 0, w: 1 },
          position: { x: 0, y: 0, z: 0 },
        },
        scale: { x: 15, y: 15, z: 15 },
        color: { r: 1, g: 0, b: 1, a: 0.9 },
      },
    ];

    return (
      <Worldview>
        <Cubes>{markers}</Cubes>
        <Axes />
      </Worldview>
    );
    // if (!this.state.isLoaded) return <p>Loading</p>;
  }
}

export default World3D;
