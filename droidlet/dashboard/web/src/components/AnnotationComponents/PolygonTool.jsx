/*
Copyright (c) Facebook, Inc. and its affiliates.

img (Javascript Image object)
    *Loaded* image to be drawn to the canvas and traced

object (String) 
    Name of the object to be traced

submitCallback (pts) 
    Function that handles submission. Takes an array of points 
    (outline of the object)
*/

import React from "react";

class PolygonTool extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      message: "",
    };

    this.update = this.update.bind(this);
    this.drawPointsAndLines = this.drawPointsAndLines.bind(this);
    this.onClick = this.onClick.bind(this);
    this.addMaskHandler = this.addMaskHandler.bind(this);
    this.deleteMaskHandler = this.deleteMaskHandler.bind(this);
    this.changeTextHandler = this.changeTextHandler.bind(this);
    this.insertPointHandler = this.insertPointHandler.bind(this);
    this.deletePointHandler = this.deletePointHandler.bind(this);
    this.zoomIn = this.zoomIn.bind(this);
    this.zoomOut = this.zoomOut.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.keyDown = this.keyDown.bind(this);
    this.drawPoint = this.drawPoint.bind(this);
    this.drawLine = this.drawLine.bind(this);
    this.localToImage = this.localToImage.bind(this);
    this.imageToLocal = this.imageToLocal.bind(this);
    this.shiftViewBy = this.shiftViewBy.bind(this);

    // default, drawing, dragging, focus, adding, inserting, deletingMask, deletingPoint
    this.mode = this.props.mode || "default";
    this.prevMode = "default";
    this.baseMode = "default";
    this.message = "";
    this.currentMaskId = 0;
    this.isDrawingPolygon = false;
    this.lastMouse = {
      x: 0,
      y: 0,
    };
    this.points = [[]];
    this.regions = [];
    this.newMask = this.props.mode === "drawing" ? true : false;

    this.canvasRef = React.createRef();

    this.zoomPixels = 300;
    this.pointSize = 10;
    this.color = "rgba(0,0,200,0.5)"; // TODO: choose random color
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");

    this.points = this.props.masks
      ? this.props.masks
          .map((maskSet) =>
            maskSet.map((pt) => ({
              x: pt.x * this.canvas.width,
              y: pt.y * this.canvas.height,
            }))
          )
          .filter((maskSet) => maskSet) // remove empty masks
      : [[]];

    this.img = this.props.img;
    this.Offset = {
      x: 0,
      y: 0,
    };
    this.baseScale = Math.min(
      this.canvas.width / this.img.width,
      this.canvas.height / this.img.height
    );
    this.scale = this.baseScale;
    this.update();
  }

  render() {
    return (
      <div>
        <p>{this.state.message}</p>
        <div>
          <button onClick={this.addMaskHandler}>Add mask (q)</button>
          <button onClick={this.deleteMaskHandler}>D(e)lete mask</button>
          <button onClick={this.insertPointHandler}>(I)nsert point</button>
          <button onClick={this.deletePointHandler}>(R)emove point</button>
          <button onClick={this.changeTextHandler}>(M)odify labels</button>
          <button onClick={() => this.props.deleteLabelHandler()}>
            Delete (l)abel
          </button>
          <button onClick={this.zoomIn}>Zoom in (=)</button>
          <button onClick={this.zoomOut}>Zoom out (-)</button>
        </div>
        <canvas
          ref={this.canvasRef}
          width="500px"
          height="500px"
          tabIndex="0"
          onClick={this.onClick}
          onMouseMove={this.onMouseMove}
          onKeyDown={this.keyDown}
        ></canvas>
      </div>
    );
  }

  update() {
    this.resetImage("small");
    this.updateMessage();
    let focused = ["dragging", "focus"].includes(this.mode);
    this.drawPointsAndLines(focused);
    this.drawRegions(focused);
    // If "Enter" was pressed, show full mask
    if (this.lastKey === "Enter") {
      this.resetImage();
      this.drawRegions();
    }
  }

  onClick(e) {
    // Let go of dragging point
    if (this.mode === "dragging") {
      let prevMode = this.prevMode;
      this.prevMode = this.mode;
      this.mode = this.prevMode === "inserting" ? this.baseMode : prevMode;
      this.update();
      return;
    }

    // Check if point was clicked
    let hoverPointIndex = this.getPointClick();
    if (hoverPointIndex != null) {
      if (
        ["adding"].includes(this.mode) &&
        hoverPointIndex[0] !== this.currentMaskId
      ) {
        return;
      }
      if (this.mode === "inserting") {
        let newPoints = this.points[hoverPointIndex[0]].slice();
        newPoints.splice(hoverPointIndex[1], 0, newPoints[hoverPointIndex[1]]);
        this.points[hoverPointIndex[0]] = newPoints;
        this.mode = this.baseMode;
      }
      if (this.mode === "deletingPoint") {
        this.points[hoverPointIndex[0]].splice(hoverPointIndex[1], 1);
        this.mode = this.baseMode;
        this.update();
        return;
      }
      this.prevMode = this.mode;
      this.mode = "dragging";
      this.draggingIndex = hoverPointIndex;
      this.currentMaskId = hoverPointIndex[0];
      this.update();
      return;
    }

    // Add new point
    if (
      ["drawing", "adding"].includes(this.mode) &&
      (this.lastKey !== "Enter" ||
        ["drawing", "adding"].includes(this.prevMode)) // case where Enter is pressed, then "add mask"
    ) {
      this.points[this.currentMaskId].push(this.localToImage(this.lastMouse));
      this.updateZoom();
      this.update();
      this.lastKey = "Mouse";
      return;
    }

    // Focus on singular region
    let regionId = this.getRegionClick();
    if (this.mode === "default") {
      if (regionId !== -1) {
        this.prevMode = this.mode;
        this.mode = "focus";
        this.currentMaskId = regionId;
      }
      this.update();
      this.lastKey = "Mouse";
      return;
    }

    // Unfocus
    if (this.mode === "focus") {
      if (regionId === -1) {
        this.prevMode = this.mode;
        this.mode = "default";
        this.currentMaskId = -1;
      }
      this.update();
      this.lastKey = "Mouse";
      return;
    }

    // Delete mask
    if (this.mode === "deletingMask") {
      if (regionId === -1) {
        return;
      }
      if (this.points.length === 1) {
        this.props.deleteLabelHandler();
      }
      this.points.splice(regionId, 1);
      this.update();
      this.lastKey = "Mouse";
      return;
    }
  }

  keyDown(e) {
    switch (e.key) {
      case " ":
        if (this.points[this.currentMaskId].length > 0) {
          this.points[this.currentMaskId].pop();
        }
        break;
      case "w":
        this.shiftViewBy(0, 10);
        break;
      case "a":
        this.shiftViewBy(10, 0);
        break;
      case "s":
        this.shiftViewBy(0, -10);
        break;
      case "d":
        this.shiftViewBy(-10, 0);
        break;
      case "q":
        this.addMaskHandler();
        break;
      case "e":
        this.deleteMaskHandler();
        break;
      case "i":
        this.insertPointHandler();
        break;
      case "r":
        this.deletePointHandler();
        break;
      case "l":
        this.props.deleteLabelHandler();
        break;
      case "m":
        this.changeTextHandler();
        break;
      case "Enter":
        if (this.lastKey === "Enter") {
          this.lastKey = null;
          this.baseMode = "default";
          this.save();
        }
        break;
      case "~":
        this.mode = "default";
        break;
      case "Escape":
        if (
          this.points[this.currentMaskId] &&
          this.points[this.currentMaskId].length >= 3
        ) {
          this.save();
          this.mode = "default";
          this.props.exitCallback();
        }
        break;
      case "=":
        this.zoomIn();
        break;
      case "-":
        this.zoomOut();
        break;
      default:
        break;
    }
    this.lastKey = e.key;
    this.update();
  }

  addMaskHandler() {
    this.currentMaskId = this.points.length;
    this.points.push([]);
    this.prevMode = this.mode;
    this.mode = "adding";
  }

  deleteMaskHandler() {
    this.prevMode = this.mode;
    this.mode = "deletingMask";
  }

  changeTextHandler() {
    this.save();
    let rect = this.canvas.getBoundingClientRect();
    let x = this.points[0][0].x + rect.left;
    let y = this.points[0][0].y + rect.top;
    this.props.changeTextHandler(x, y);
  }

  insertPointHandler() {
    this.baseMode =
      this.points[this.currentMaskId] &&
      this.points[this.currentMaskId].length === 0
        ? "default"
        : this.mode;
    this.prevMode = this.mode;
    this.mode = "inserting";
  }

  deletePointHandler() {
    this.baseMode = this.mode;
    this.prevMode = this.mode;
    this.mode = "deletingPoint";
  }

  zoomIn() {
    this.zoomPixels -= 10;
    this.updateZoom();
  }

  zoomOut() {
    this.zoomPixels += 10;
    this.updateZoom();
  }

  updateMessage() {
    let newMessage = "";
    switch (this.mode) {
      case "adding":
        newMessage = "Please add the new mask";
        break;
      case "deletingMask":
        newMessage = "Select which mask to delete";
        break;
      case "inserting":
        newMessage = "Select a point to duplicate";
        break;
      case "deletingPoint":
        newMessage = "Select a point to delete";
        break;
      default:
        newMessage = "Please trace the " + (this.props.object || "object");
        break;
    }
    if (newMessage !== this.state.message) {
      this.setState({ message: newMessage });
    }
  }

  updateZoom() {
    this.scale = Math.min(
      this.canvas.width / this.zoomPixels,
      this.canvas.height / this.zoomPixels
    );
    if (
      this.currentMaskId === -1 ||
      !this.points[this.currentMaskId] ||
      this.points[this.currentMaskId].length === 0 ||
      !["default", "dragging", "drawing", "adding"].includes(this.mode)
    ) {
      return;
    }
    let points = this.points[this.currentMaskId];
    this.Offset = {
      x: -(points[points.length - 1].x - this.zoomPixels / 2) * this.scale,
      y: -(points[points.length - 1].y - this.zoomPixels / 2) * this.scale,
    };
  }

  onMouseMove(e) {
    var rect = this.canvas.getBoundingClientRect();
    this.lastMouse = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top + 1,
    };
    if (this.mode === "dragging") {
      this.points[this.draggingIndex[0]][this.draggingIndex[1]] =
        this.localToImage(this.lastMouse);
    }
    this.update();
  }

  save() {
    this.props.submitCallback(
      this.points.map((pts) =>
        pts.map((p) => ({
          x: p.x / this.canvas.width,
          y: p.y / this.canvas.height,
        }))
      ),
      this.newMask
    );
  }

  /***************************************************************************
   * Utilities
   ***************************************************************************/

  getPointClick() {
    for (let i = 0; i < this.points.length; i++) {
      for (let j = 0; j < this.points[i].length; j++) {
        if (
          this.distance(this.points[i][j], this.localToImage(this.lastMouse)) <
          this.pointSize / 2
        ) {
          return [i, j];
        }
      }
    }
    return null;
  }

  getRegionClick() {
    let regionId = -1;
    for (let i = 0; i < this.regions.length; i++) {
      if (
        this.regions[i] &&
        this.ctx.isPointInPath(
          this.regions[i],
          this.lastMouse.x,
          this.lastMouse.y
        )
      ) {
        regionId = i;
      }
    }
    return regionId;
  }

  drawPointsAndLines(focus = false) {
    for (let i = 0; i < this.points.length; i++) {
      // Continue if focusing on specific mask and id isn't equal
      if (focus === true && i !== this.currentMaskId) continue;
      // Continue if mask is empty
      if (!this.points[i] || this.points[i].length === 0) continue;

      // Points and Lines
      for (let j = 0; j < this.points[i].length - 1; j++) {
        this.drawLine(this.points[i][j], this.points[i][j + 1]);
        this.drawPoint(this.points[i][j]);
      }
      this.drawPoint(this.points[i][this.points[i].length - 1]); // Final point
      // Line connecting start to finish
      if (
        !["drawing", "adding"].includes(this.mode) &&
        !["drawing", "adding"].includes(this.prevMode)
      ) {
        this.drawLine(
          this.points[i][0],
          this.points[i][this.points[i].length - 1]
        );
      }
    }
    // Line to mouse
    if (
      this.points[this.currentMaskId] &&
      this.points[this.currentMaskId].length > 0 &&
      ["drawing", "adding"].includes(this.mode)
    ) {
      this.drawLine(
        this.points[this.currentMaskId][
          this.points[this.currentMaskId].length - 1
        ],
        this.localToImage(this.lastMouse)
      );
    }
  }

  drawRegions(focus = false) {
    this.regions = [];
    for (let i = 0; i < this.points.length; i++) {
      // Continue if focusing on specific mask and id isn't equal
      if (focus && i !== this.currentMaskId) continue;
      let region = this.drawRegion(this.points[i]);
      this.regions.push(region);
    }
  }

  drawPoint(pt) {
    this.ctx.fillStyle = "black";
    if (
      this.distance(pt, this.localToImage(this.lastMouse)) <
      this.pointSize / 2
    ) {
      this.ctx.fillStyle = "green";
    }
    this.ctx.fillRect(pt.x - 2.5, pt.y - 2.5, 5, 5);
  }

  drawLine(pt1, pt2) {
    this.ctx.beginPath();
    this.ctx.moveTo(pt1.x, pt1.y);
    this.ctx.lineTo(pt2.x, pt2.y);
    this.ctx.stroke();
  }

  drawRegion(points) {
    if (!points || points.length < 3) {
      return;
    }
    let region = new Path2D();
    region.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      region.lineTo(points[i].x, points[i].y);
    }
    region.closePath();
    this.ctx.fillStyle = this.color;
    this.ctx.fill(region, "evenodd");

    return region;
  }

  localToImage(pt) {
    let newX = (pt.x - this.Offset.x) / this.scale;
    let newY = (pt.y - this.Offset.y) / this.scale;
    return {
      x: Math.min(Math.max(newX, 0), 512), // 512 is width/heigh of image on right
      y: Math.min(Math.max(newY, 0), 512), // this.canvas.width is 500 for some reason
    };
  }

  imageToLocal(pt) {
    return {
      x: pt.x * this.scale + this.Offset.x,
      y: pt.y * this.scale + this.Offset.y,
    };
  }

  resetView() {
    this.Offset = {
      x: 0,
      y: 0,
    };
    this.scale = Math.min(
      this.canvas.width / this.img.width,
      this.canvas.height / this.img.height
    );
  }

  resetImage(type = "full") {
    // full, small
    this.ctx.resetTransform();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (type === "full") {
      this.ctx.setTransform(this.baseScale, 0, 0, this.baseScale, 0, 0);
    } else if (type === "small") {
      this.ctx.setTransform(
        this.scale,
        0,
        0,
        this.scale,
        this.Offset.x,
        this.Offset.y
      );
    }
    this.ctx.drawImage(this.img, 0, 0);
  }

  shiftViewBy(x, y) {
    this.Offset = {
      x: this.Offset.x + x,
      y: this.Offset.y + y,
    };
  }

  distance(pt1, pt2) {
    return Math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2);
  }

  // Not used. Meant for highlighting segments
  distanceToSegment(pt, p1, p2) {
    let d = this.distance(p1, p2);
    if (d === 0) {
      return this.distance(pt, p1);
    }
    let t = ((pt.x - p1.x) * (p2.x - p1.x) + (pt.y - p1.y) * (p2.y - p1.y)) / d;
    t = Math.max(0, Math.min(1, t));
    let proj = {
      x: p1.x + t * (p2.x - p1.x),
      y: p1.y + t * (p2.y - p1.y),
    };
    return this.distance(pt, proj);
  }
}

export default PolygonTool;
