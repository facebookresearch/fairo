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
    this.onClick = this.onClick.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.keyDown = this.keyDown.bind(this);
    this.drawPoint = this.drawPoint.bind(this);
    this.drawLine = this.drawLine.bind(this);
    this.localToImage = this.localToImage.bind(this);
    this.imageToLocal = this.imageToLocal.bind(this);
    this.shiftViewBy = this.shiftViewBy.bind(this);

    this.mode = "default"; // default, dragging (could implement: draw, erase, duplicate, select, delete, etc)
    this.prevMode = "default";
    this.isDrawingPolygon = false;
    this.lastMouse = {
      x: 0,
      y: 0,
    };
    this.points = [];

    this.canvasRef = React.createRef();

    this.zoomPixels = 300;
    this.pointSize = 10;
    this.color = "rgba(0,0,200,0.5)"; // TODO: choose random color
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");

    this.points = this.props.masks.map((maskSet) =>
      maskSet.map((pt) => ({
        x: pt.x * this.canvas.width,
        y: pt.y * this.canvas.height,
      }))
    );
    console.log("shoudl be in array with objects with x and y", this.points);

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
    if (this.lastKey !== "Enter") {
      for (let i = 0; i < this.points.length; i++) {
        // Lines
        for (let j = 0; j < this.points[i].length - 1; j++) {
          this.drawLine(this.points[i][j], this.points[i][j + 1]);
        }
        // Line to mouse
        if (this.points[i].length > 0 && ["drawing"].includes(this.mode)) {
          this.drawLine(
            this.points[i][this.points[i].length - 1],
            this.localToImage(this.lastMouse)
          );
        }
        // Draw points
        this.points.forEach((ptSet) => {
          ptSet.forEach((pt) => {
            this.drawPoint(pt);
          });
        });
      }
    } else {
      this.ctx.resetTransform();
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.setTransform(this.baseScale, 0, 0, this.baseScale, 0, 0);
      this.ctx.drawImage(this.img, 0, 0);
      if (this.points.length < 3) {
        return;
      }
      let region = new Path2D();
      region.moveTo(this.points[0].x, this.points[0].y);
      for (let i = 1; i < this.points.length; i++) {
        region.lineTo(this.points[i].x, this.points[i].y);
      }
      region.closePath();
      this.ctx.fillStyle = "rgba(0,200,0,.5)";
      this.ctx.fill(region, "evenodd");
    }
  }

  updateZoom() {
    this.scale = Math.min(
      this.canvas.width / this.zoomPixels,
      this.canvas.height / this.zoomPixels
    );
    if (this.points.length === 0) {
      return;
    }
    this.Offset = {
      x:
        -(this.points[this.points.length - 1].x - this.zoomPixels / 2) *
        this.scale,
      y:
        -(this.points[this.points.length - 1].y - this.zoomPixels / 2) *
        this.scale,
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

  onClick(e) {
    if (this.mode === "dragging") {
      this.prevMode = this.mode;
      this.mode = "default";
      this.update();
      return;
    }

    // Check if point was clicked
    let hoverPointIndex = null;
    for (let i = 0; i < this.points.length; i++) {
      for (let j = 0; j < this.points[i].length; j++) {
        if (
          this.distance(this.points[i][j], this.localToImage(this.lastMouse)) <
          this.pointSize / 2
        ) {
          hoverPointIndex = [i, j];
        }
      }
    }
    if (hoverPointIndex != null) {
      console.log("dragging now");
      this.prevMode = this.mode;
      this.mode = "dragging";
      this.draggingIndex = hoverPointIndex;
      this.update();
      return;
    }

    if (this.lastKey !== "Enter" && this.mode === "drawing") {
      this.points.push(this.localToImage(this.lastMouse));
    }
    this.updateZoom();
    this.update();
    this.lastKey = "Mouse";
  }

  keyDown(e) {
    switch (e.key) {
      case " ":
        if (this.points.length > 0) {
          this.points.pop();
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
            this.points.map((p) => ({
              x: p.x / this.canvas.width,
              y: p.y / this.canvas.height,
            }))
          );
        }
        break;
      case "=":
        this.zoomPixels -= 10;
        this.updateZoom();
        break;
      default:
        break;
    }
    this.lastKey = e.key;
    this.update();
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
