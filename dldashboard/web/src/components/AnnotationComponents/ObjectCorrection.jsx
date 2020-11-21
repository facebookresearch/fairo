/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import DataEntry from "./DataEntry";
import MaskCorrectionTool from "./AnnotationComponents/MaskCorrectionTool";
import turk from "./turk.js";

class ObjectCorrection extends React.Component {
  constructor(props) {
    super(props);

    this.propertyMap = {};
    this.pointMap = {};
    this.nameMap = {};
    this.objectIds = [];

    this.state = {
      currentIndex: 0,
      phase: "labeling",
    };

    this.drawingFinished = this.drawingFinished.bind(this);
  }

  componentDidMount() {
    this.canvas = document.getElementById("main-canvas");
    this.ctx = this.canvas.getContext("2d");

    this.image = new Image();
    this.image.onload = () => {
      this.scale = Math.min(
        this.canvas.width / this.image.width,
        this.canvas.height / this.image.height
      );
      this.update();
      this.forceUpdate();
    };
    this.image.src = this.props.imgUrl;
  }

  render() {
    let currentItem = this.props.targets[this.state.currentIndex];
    console.log(currentItem);
    console.log(this.state.currentIndex);

    if (this.state.phase == "labeling") {
      return (
        <div>
          <p>
            Please verify that the outlined object matches with the data in the
            popup, and correct any errors.
          </p>
          <canvas id="main-canvas" width="600px" height="600px"></canvas>
          <DataEntry
            x={(currentItem.bbox[0] + currentItem.bbox[2]) * this.scale + 10}
            y={currentItem.bbox[1] * this.scale + 50}
            defaultName={currentItem.name}
            defaultProperties={currentItem.props}
            onSubmit={(data) => {
              this.currentData = data;
              this.setState({
                phase: "drawing",
              });
            }}
          ></DataEntry>
        </div>
      );
    }

    if (this.state.phase == "drawing") {
      return (
        <MaskCorrectionTool
          img={this.image}
          object={this.currentData.name}
          bbox={currentItem.bbox}
          submitCallback={this.drawingFinished}
        ></MaskCorrectionTool>
      );
    }
  }

  update() {
    let currentItem = this.props.targets[this.state.currentIndex];

    this.ctx.resetTransform();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
    //Draw image scaled and repostioned
    this.ctx.drawImage(this.image, 0, 0);
    //Draw box around target
    this.ctx.beginPath();
    this.ctx.strokeStyle = "rgba(255,0,0,.5)";
    this.ctx.lineWidth = 10;
    this.ctx.rect(...currentItem.bbox);
    this.ctx.stroke();
  }

  drawingFinished(data) {
    this.propertyMap[this.state.currentIndex] = this.currentData.tags;
    this.pointMap[this.state.currentIndex] = data;
    this.nameMap[this.state.currentIndex] = this.currentData.name;
    this.objectIds.push(this.state.currentIndex);

    if (this.state.currentIndex + 1 == this.props.targets.length) {
      this.submit();
    }

    this.setState({
      phase: "labeling",
      currentIndex: this.state.currentIndex + 1,
    });
  }

  submit() {
    turk.submit({
      objectIds: this.objectIds,
      properties: this.propertyMap,
      points: this.pointMap,
      names: this.nameMap,
      metaData: {
        width: this.image.width,
        height: this.image.height,
      },
    });
  }
}

export default ObjectCorrection;
