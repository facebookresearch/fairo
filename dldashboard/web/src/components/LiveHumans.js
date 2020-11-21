/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LiveHumans.js

import React from "react";
import { Rnd } from "react-rnd";
import { Stage, Layer, Image as KImage, Line } from "react-konva";
import { schemeTableau10 as colorScheme } from "d3-scale-chromatic";

const draw_lines = [
  ["left_ear", "left_eye"],
  ["right_ear", "right_eye"],
  ["left_eye", "right_eye"],
  ["left_eye", "nose"],
  ["right_eye", "nose"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["left_shoulder", "left_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_shoulder", "right_hip"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["right_hip", "left_hip"],
  ["right_shoulder", "left_shoulder"],
];

class LiveHumans extends React.Component {
  constructor(props) {
    super(props);
    this.onResize = this.onResize.bind(this);
    this.initialState = {
      height: props.height,
      width: props.width,
      rgb: null,
      humans: null,
    };
    this.state = this.initialState;
  }

  onResize(e, direction, ref, delta, position) {
    this.setState({
      width: parseInt(ref.style.width, 10),
      height: parseInt(ref.style.height, 10),
    });
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    const { height, width, rgb, humans } = this.state;
    const { offsetW, offsetH } = this.props;

    if (rgb === null) {
      return (
        <Rnd
          default={{
            x: offsetW,
            y: offsetH,
            width: width,
            height: height,
          }}
          lockAspectRatio={true}
          onResize={this.onResize}
        >
          <p>Loading...</p>
        </Rnd>
      );
    }

    /* example "humans": [
        {"xyz": [0.16, -0.08, 0.79], 
        "keypoints": {
            "nose": [457, 156, 0],
            "left_eye": [458, 152, 0],
            "right_eye": [488, 158, 0],
            "left_ear": [503, 177, 0],
            "right_ear": [460, 167, 0],
            "left_shoulder": [511, 64, 0],
            "right_shoulder": [511, 68, 0],
            "left_elbow": [470, 158, 0],
            "right_elbow": [470, 158, 0],
            "left_wrist": [496, 215, 0],
            "right_wrist": [508, 226, 0],
            "left_hip": [511, 54, 0],
            "right_hip": [511, 208, 0],
            "left_knee": [466, 159, 0],
            "right_knee": [469, 158, 0],
            "left_ankle": [507, 235, 0],
            "right_ankle": [500, 223, 0]
        }
    }]
    */
    var renderedHumans = [];
    let i = 0,
      j = 0;
    humans.forEach((human) => {
      let scale = height / 512;
      let color = colorScheme[i++];
      if (i === colorScheme.length) {
        i = 0;
      }
      draw_lines.forEach((row) => {
        let p1 = human.keypoints[row[0]];
        let p2 = human.keypoints[row[1]];
        let x1 = parseInt(p1[0] * scale),
          y1 = parseInt(p1[1] * scale);
        let x2 = parseInt(p2[0] * scale),
          y2 = parseInt(p2[1] * scale);
        let points = [x1, y1, x2, y2];
        renderedHumans.push(<Line key={j++} points={points} stroke={color} />);
      });
    });

    return (
      <Rnd
        default={{
          x: offsetW,
          y: offsetH,
          width: width,
          height: height,
        }}
        lockAspectRatio={true}
        onResize={this.onResize}
      >
        <Stage width={width} height={height}>
          <Layer>
            <KImage image={rgb} width={width} height={height} />
            {renderedHumans}
          </Layer>
        </Stage>
      </Rnd>
    );
  }
}

export default LiveHumans;
