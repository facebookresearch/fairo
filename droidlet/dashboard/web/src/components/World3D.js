/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/World3D.js

import React from "react";
import Worldview, {
  Points,
  Cylinders,
  Cones,
  Cubes,
  Axes,
  Arrows,
  Lines,
} from "regl-worldview";

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
      points: [],
      colors: [],
      base: false,
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
    const { isLoaded, all_points, all_colors, points, colors, base } =
      this.state;
    if (!isLoaded) return null;

    console.log(base);
    const x = base[0];
    const y = base[1];
    const yaw = base[2];
    const points_marker = {
      points: points,
      scale: { x: 3, y: 3, z: 3 },
      colors: colors,
      pose: {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    };

    const cone_marker = {
      pose: {
        orientation: { x: 0, y: 0, z: 0, w: 1 },
        position: { x: y, y: -x, z: 1 },
      },
      scale: { x: 0.1, y: 0.1, z: 1 },
      color: { r: 0, g: 1, b: 1, a: 1 },
    };

    const line_marker = {
      closed: false,
      pose: {
        position: { x: y, y: -x, z: 1 },
        orientation: { x: 0, y: 0, z: yaw, w: 1 },
      },
      scaleInvariant: false,
      scale: { x: 0.1, y: 0.1, z: 0.1 },
      points: [
        [0, 0, 0],
        [0.1, 0.1, 0.1],
      ],
      color: { r: 1, g: 0, b: 1, a: 1 },
    };

    const camera_state = {
      distance: 10,
      target: [y, -x, 1],
      targetOrientation: [0, 0, yaw, 1],
      perspective: true,
      phi: Math.PI / 3,
      thetaOffset: Math.PI / 3,
    };

    return (
      <Worldview defaultCameraState={camera_state}>
        <Points>{[points_marker]}</Points>
        <Cylinders>{[cone_marker]}</Cylinders>
        <Lines>{[line_marker]}</Lines>
        <Axes />
      </Worldview>
    );
    // if (!this.state.isLoaded) return <p>Loading</p>;
  }
}

export default World3D;
