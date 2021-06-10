/*
Copyright (c) Facebook, Inc. and its affiliates.

Specifies a react component that takes in an image url, 
and provides a annotation UI for tagging and segmenting objects in an image

props:

imgUrl: url of the image to annotate

masks: array of points for the masks
*/

import React from "react";
import DataEntry from "./DataEntry";
import PolygonTool from "./PolygonTool";
import SegmentRenderer from "./SegmentRenderer";

const COLORS = [
  "rgba(0,200,0,.5)",
  "rgba(200,0,0,.5)",
  "rgba(0,0,200,.5)",
  "rgba(200,200,0,.5)",
  "rgba(0,200,200,.5)",
  "rgba(200,0,200,.5)",
  "rgba(150,50,50,.5)",
  "rgba(255, 153, 0, .5)",
  "rgba(128,0,128,.5)",
  "rgba(0,204,255,.5)",
  "rgba(153,204,0,.5)",
];

class ObjectAnnotation extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      objectIds: [...Array(this.props.objects.length).keys()], // [0, ..., maskLength-1]
      currentMode: "select", // one of select, fill_data, draw
      currentOverlay: null,
      currentMask: null,
    };

    this.currentId = this.props.objects.length;
    this.nameMap = {};
    this.pointMap = {};
    this.propertyMap = {};
    for (let i = 0; i < this.props.objects.length; i++) {
      let curObject = this.props.objects[i];
      this.nameMap[i] = curObject.label;
      this.pointMap[i] = curObject.mask;
      this.propertyMap[i] = curObject.properties;
    }

    this.registerClick = this.registerClick.bind(this);

    if (this.props.image !== undefined) {
      this.image = this.props.image;
    } else {
      this.image = new Image();
      this.image.onload = () => {
        this.forceUpdate();
      };
      this.image.src = this.props.imgUrl;
    }
    this.overtime = false;
    setInterval(() => {
      //alert("Please finish what you're working on and click Submit Task below")
      this.overtime = true;
    }, 1000 * 60 * window.MINUTES);
  }

  render() {
    if (this.state.currentMode !== "draw_polygon") {
      return (
        <div>
          <p>
            Label and outline as <b>many objects as possible.</b> Click an
            object in the image to start. {this.state.objectIds.length}{" "}
            object(s) labeled.
          </p>
          {this.state.currentOverlay}
          <SegmentRenderer
            img={this.image}
            objects={this.state.objectIds}
            pointMap={this.pointMap}
            colors={COLORS}
            onClick={this.registerClick}
          />
          <button onClick={this.submit.bind(this)}>
            Finished annotating objects
          </button>
        </div>
      );
    } else {
      return (
        <PolygonTool
          img={this.image}
          object={this.drawing_data.name}
          masks={this.pointMap[this.state.currentMask]}
          exitCallback={() => {
            this.setState({ currentMode: "select" });
          }}
          submitCallback={this.drawingFinished.bind(this)}
        ></PolygonTool>
      );
    }
  }

  registerClick(x, y, regionFound, region) {
    if (this.state.currentMode === "select") {
      if (regionFound) {
        this.drawing_data = {
          tags: this.propertyMap[region],
          name: this.nameMap[region],
        };
        this.setState({
          currentMode: "draw_polygon",
          currentOverlay: null,
          currentMask: region,
        });
      } else {
        // Build overlay component
        var overlay = (
          <DataEntry x={x} y={y} onSubmit={this.dataEntered.bind(this)} />
        );
        // Update State
        this.setState({
          currentMode: "fill_data",
          currentOverlay: overlay,
          currentMask: -1,
        });
      }
    }
  }

  dataEntered(objectData) {
    this.drawing_data = objectData;
    this.setState({
      currentMode: "draw_polygon",
      currentOverlay: null,
    });
  }

  drawingFinished(data) {
    this.propertyMap[this.currentId] = this.drawing_data.tags;
    this.pointMap[this.currentId] = [data];
    // HOLLIS NOTE: data in form of [{x: 0, y: 0}, {x: 0, y: 0}, ...]
    // Need to implement functionality for multiple masks for the same label
    this.nameMap[this.currentId] = this.drawing_data.name;
    this.setState(
      {
        currentMode: "select",
        objectIds: this.state.objectIds.splice(0).concat(this.currentId),
      },
      () => {
        this.currentId += 1;
      }
    );
  }

  submit() {
    if (this.state.objectIds.length < window.MIN_OBJECTS) {
      alert("Label more objects, or your HIT may be rejected");
      return;
    }

    const postData = {
      nameMap: this.nameMap,
      propertyMap: this.propertyMap,
      pointMap: this.pointMap,
    };
    this.props.stateManager.socket.emit("saveObjectAnnotation", postData);
    if (this.props.not_turk === true) return;

    // TODO: uncomment this to get working in a turk setting again
    // import turk from '../turk'
    // turk.submit(
    //     {objectIds: this.state.objectIds,
    //     properties: this.propertyMap,
    //     points: this.pointMap,
    //     names: this.nameMap,
    //     metaData: {
    //         width: this.image.width,
    //         height: this.image.height
    //     }
    //     }
    // )
  }
}

export default ObjectAnnotation;
