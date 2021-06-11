/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

/* Props

img (Javascript Image object)
    *Loaded* image to be drawn to the canvas and traced

object (String) 
    Name of the object to be traced

submitCallback (pts) 
    Function that handles submission. Takes an array of points 
    (outline of the object)
*/

class PolygonTool extends React.Component {
  constructor(props) {
    super(props);

    this.update = this.update.bind(this);
    this.drawPointsAndLines = this.drawPointsAndLines.bind(this);
    this.onClick = this.onClick.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.keyDown = this.keyDown.bind(this);
    this.drawPoint = this.drawPoint.bind(this);
    this.drawLine = this.drawLine.bind(this);
    this.localToImage = this.localToImage.bind(this);
    this.imageToLocal = this.imageToLocal.bind(this);
    this.shiftViewBy = this.shiftViewBy.bind(this);

    this.mode = this.props.mode || "default"; // default, drawing, dragging, focus (could implement: draw, erase, duplicate, select, delete, etc)
    this.prevMode = "default";
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
      ? this.props.masks.map((maskSet) =>
          maskSet.map((pt) => ({
            x: pt.x * this.canvas.width,
            y: pt.y * this.canvas.height,
          }))
        )
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
        <p>Please trace the {this.props.object}</p>
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
    // Draw image scaled and repostioned
    this.ctx.resetTransform();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.setTransform(
      this.scale,
      0,
      0,
      this.scale,
      this.Offset.x,
      this.Offset.y
    );
    this.ctx.drawImage(this.img, 0, 0);

    // Draw points and lines
    if (["default", "drawing", "dragging", "focus"].includes(this.mode)) {
      this.drawPointsAndLines(["dragging", "focus"].includes(this.mode));
    }

    // Draw regions
    if (["default", "focus"].includes(this.mode)) {
      this.drawRegions(["dragging", "focus"].includes(this.mode));
    }
  }

  onClick(e) {
    // Let go of dragging point
    if (this.mode === "dragging") {
      let prevMode = this.prevMode;
      this.prevMode = this.mode;
      this.mode = prevMode;
      console.log("updating mode from", this.prevMode, "to", this.mode);
      this.update();
      return;
    }

    // Check if point was clicked
    let hoverPointIndex = this.getPointClick();
    if (hoverPointIndex != null) {
      this.prevMode = this.mode;
      this.mode = "dragging";
      console.log("updating mode from", this.prevMode, "to", this.mode);
      this.draggingIndex = hoverPointIndex;
      this.currentMaskId = hoverPointIndex[0];
      this.update();
      return;
    }

    // Add new point
    if (this.lastKey !== "Enter" && this.mode === "drawing") {
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
        console.log("updating mode from", this.prevMode, "to", this.mode);
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
        console.log("updating mode from", this.prevMode, "to", this.mode);
        this.currentMaskId = -1;
      }
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
      case "Enter":
        if (this.lastKey === "Enter") {
          this.lastKey = null;
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
        break;
      case "~":
        this.mode = "default";
        break;
      case "Escape":
        this.props.submitCallback(
          this.points.map((pts) =>
            pts.map((p) => ({
              x: p.x / this.canvas.width,
              y: p.y / this.canvas.height,
            }))
          ),
          this.newMask
        );
        this.mode = "default";
        this.props.exitCallback();
        break;
      case "=":
        this.zoomPixels -= 10;
        this.updateZoom();
        break;
      case "-":
        this.zoomPixels += 10;
        this.updateZoom();
        break;
      default:
        break;
    }
    this.lastKey = e.key;
    this.update();
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
      !["default", "dragging"].includes(this.mode)
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
      if (this.points[i].length === 0) continue;
      // Points and Lines
      for (let j = 0; j < this.points[i].length - 1; j++) {
        this.drawLine(this.points[i][j], this.points[i][j + 1]);
        this.drawPoint(this.points[i][j]);
      }
      this.drawPoint(this.points[i][this.points[i].length - 1]); // Final point
      // Line to mouse
      if (this.points[i].length > 0 && ["drawing"].includes(this.mode)) {
        this.drawLine(
          this.points[i][this.points[i].length - 1],
          this.localToImage(this.lastMouse)
        );
      }
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
    if (points.length < 3) {
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
    return {
      x: (pt.x - this.Offset.x) / this.scale,
      y: (pt.y - this.Offset.y) / this.scale,
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

  shiftViewBy(x, y) {
    this.Offset = {
      x: this.Offset.x + x,
      y: this.Offset.y + y,
    };
  }

  distance(pt1, pt2) {
    return Math.max(Math.abs(pt1.x - pt2.x), Math.abs(pt1.y - pt2.y)) * 2;
  }
}

export default PolygonTool;
