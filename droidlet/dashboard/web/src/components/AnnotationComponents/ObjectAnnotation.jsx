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
      currentMode: "select", // one of select, fill_data, draw_polygon, start_polygon
      currentOverlay: null,
      currentMaskId: null,
    };

    this.nextId = this.props.objects.length;
    this.nameMap = {};
    this.pointMap = {};
    this.propertyMap = {};
    for (let i = 0; i < this.props.objects.length; i++) {
      let curObject = this.props.objects[i];
      this.nameMap[i] = curObject.label;
      this.pointMap[i] = curObject.mask;
      for (let j in this.pointMap[i]) {
        if (this.pointMap[i][j].length < 3) {
          delete this.pointMap[i][j];
        }
      }
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
    if (["draw_polygon", "start_polygon"].includes(this.state.currentMode)) {
      return (
        <PolygonTool
          img={this.image}
          object={this.drawing_data.name}
          masks={this.pointMap[this.state.currentMaskId]}
          color={COLORS[this.state.objectIds.length % COLORS.length]}
          exitCallback={() => {
            this.setState({ currentMode: "select" });
          }}
          submitCallback={this.drawingFinished.bind(this)}
          deleteLabelHandler={this.deleteLabelHandler.bind(this)}
          changeTextHandler={this.changeTextHandler.bind(this)}
          mode={this.state.currentMode === "start_polygon" ? "drawing" : null}
        ></PolygonTool>
      );
    } else {
      return (
        <div>
          <p>
            Label and outline as <b>many objects as possible.</b> Click an
            object in the image to start. {this.state.objectIds.length}{" "}
            object(s) labeled.
          </p>
          {this.state.currentOverlay}
          <div>
            {this.state.objectIds.map((id, i) => (
              <button
                key={id}
                style={{ backgroundColor: COLORS[i % COLORS.length] }}
                onClick={() => this.labelSelectHandler(id)}
              >
                {this.nameMap[id]}
              </button>
            ))}
          </div>
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
    }
  }

  drawingFinished(data, newMask) {
    this.pointMap[this.state.currentMaskId] = data;
    this.setState({
      currentMode: "select",
      objectIds: newMask
        ? this.state.objectIds.splice(0).concat(this.state.currentMaskId)
        : this.state.objectIds,
    });
    if (newMask) {
      var overlay = (
        <DataEntry
          x={this.clickPoint.x}
          y={this.clickPoint.y}
          onSubmit={this.dataEntrySubmit.bind(this)}
        />
      );
      this.setState({
        currentMode: "fill_data",
        currentOverlay: overlay,
      });
    }
  }

  deleteLabelHandler() {
    delete this.nameMap[this.state.currentMaskId];
    delete this.pointMap[this.state.currentMaskId];
    delete this.propertyMap[this.state.currentMaskId];
    let newObjectIds = this.state.objectIds.slice();
    let index = this.state.objectIds.indexOf(
      parseInt(this.state.currentMaskId)
    );
    if (index >= 0) {
      newObjectIds.splice(index, 1);
    }
    this.setState({
      currentMode: "select",
      currentMaskId: -1,
      objectIds: newObjectIds,
    });
  }

  changeTextHandler(x, y) {
    var overlay = (
      <DataEntry
        x={x}
        y={y}
        onSubmit={this.dataEntrySubmit.bind(this)}
        label={this.drawing_data.name}
        tags={this.drawing_data.tags}
      />
    );
    this.setState({
      currentMode: "fill_data",
      currentOverlay: overlay,
    });
  }

  dataEntrySubmit(objectData) {
    this.drawing_data = objectData;
    this.propertyMap[this.state.currentMaskId] = this.drawing_data.tags;
    this.nameMap[this.state.currentMaskId] = this.drawing_data.name;
    this.setState({
      currentMode: "select",
      currentOverlay: null,
    });
  }

  labelSelectHandler(id) {
    this.setState({
      currentMode: "draw_polygon",
      currentOverlay: null,
      currentMaskId: id,
    });
    this.drawing_data = {
      tags: this.propertyMap[id],
      name: this.nameMap[id],
    };
  }

  registerClick(x, y, regionFound, regionId) {
    if (this.state.currentMode === "select") {
      if (regionFound) {
        this.drawing_data = {
          tags: this.propertyMap[regionId],
          name: this.nameMap[regionId],
        };
        this.setState({
          currentMode: "draw_polygon",
          currentOverlay: null,
          currentMaskId: regionId,
        });
      } else if (this.state.currentMode !== "fill_data") {
        this.drawing_data = {
          tags: null,
          name: null,
        };
        this.setState({
          currentMode: "start_polygon",
          currentMaskId: this.nextId,
        });
        this.clickPoint = { x, y };
        this.nextId += 1;
      }
    }
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
