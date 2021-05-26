/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// src/components/LiveImage.js

import React from "react";
import { Rnd } from "react-rnd";
import { Stage, Layer, Image as KImage } from "react-konva";

class LiveImage extends React.Component {
  constructor(props) {
    super(props);
    this.onResize = this.onResize.bind(this);
    this.initialState = {
      height: props.height,
      width: props.width,
      rgb: null,
      depth: null,
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
    const { height, width, rgb, depth } = this.state;
    const { offsetW, offsetH, isMobile } = this.props;

    let img = rgb;

    if (rgb === null && depth == null) {
      if (isMobile) {
        return <p>Loading...</p>;
      }
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
    } else if (this.props.type === "depth") {
      img = depth;
    }
    if (isMobile) {
      return (
        <Stage width={width} height={height}>
          <Layer>
            <KImage image={img} width={width} height={height} />
          </Layer>
        </Stage>
      );
    }
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
            <KImage image={img} width={width} height={height} />
          </Layer>
        </Stage>
      </Rnd>
    );
  }
}

export default LiveImage;
