/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

/* Polygon tool adpated to work with video inputs */

/* Props

img (OffscreenCanvas)

object (String) 
    Name of the object to be traced

submitCallback (pts) 
    Function that handles submission. Takes an array of points

startPoints (pts)
    A list of preexisting points for the editing mask use case
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

    this.isDrawingPolygon = true;
    this.lastMouse = {
      x: 0,
      y: 0,
    };
    this.dragging = false;

    this.canvasRef = React.createRef();

    this.zoomPixels = 400;
    this.pointSize = 20;
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");

    this.points = this.props.startPoints;

    this.img = this.props.img;
    this.Offset = {
      x: 0,
      y: 0,
    };
    this.baseScale = Math.min(
      this.canvas.width / this.img.videoWidth,
      this.canvas.height / this.img.videoHeight
    );
    this.scale = this.baseScale;

    if (this.points.length >= 1) {
      let bbox = this.calculateBbox();
      this.zoomPixels = Math.max(bbox[2], bbox[3]) + 40;
      this.updateZoom();
      this.Offset = {
        x: -(bbox[0] - 20) * this.scale,
        y: -(bbox[1] - 20) * this.scale,
      };
    }

    this.update();
  }

  render() {
    return (
      <div>
        <p>
          Click the <b>edge</b> of the {this.props.object} to start.
        </p>
        <canvas
          ref={this.canvasRef}
          width="700px"
          height="700px"
          tabIndex="0"
          onClick={this.onClick}
          onMouseMove={this.onMouseMove}
          onKeyDown={this.keyDown}
        ></canvas>
        <div>
          <p>[SPACE] to undo</p>
          <p>[WASD] to move around the image</p>
          <p>[-/+] to zoom in and out</p>
          <p>[Enter] press once to preview, and twice to submit</p>
          <p>[CLICK] to draw a new point, or edit an old one</p>
        </div>
      </div>
    );
  }

  update() {
    //clear and transform
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
    //Draw image scaled and repostioned
    this.ctx.drawImage(this.img, 0, 0);
    //Draw points and lines
    if (this.lastKey != "Enter") {
      for (let i = 0; i < this.points.length - 1; i++) {
        this.drawLine(this.points[i], this.points[i + 1]);
      }
      if (this.points.length > 0 && !this.dragging) {
        this.drawLine(
          this.points[this.points.length - 1],
          this.localToImage(this.lastMouse)
        );
      }

      this.points.map((pt) => {
        this.drawPoint(pt);
      });
    } else {
      this.ctx.resetTransform();
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
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
      this.canvas.width / this.img.videoWidth,
      this.canvas.height / this.img.height
    );
    this.scale = Math.min(
      this.canvas.width / this.zoomPixels,
      this.canvas.height / this.zoomPixels
    );
    if (this.points.length == 0) {
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
    if (this.dragging) {
      this.points[this.draggingIndex] = this.localToImage(this.lastMouse);
    }
    this.update();
  }

  onClick(e) {
    if (this.dragging) {
      this.dragging = false;
      this.update();
      return;
    }

    let hoverPointIndex = null;
    for (let i = 0; i < this.points.length; i++) {
      if (
        this.distance(this.points[i], this.localToImage(this.lastMouse)) <
        this.pointSize / 2
      ) {
        hoverPointIndex = i;
      }
    }

    if (hoverPointIndex != null) {
      this.dragging = true;
      this.draggingIndex = hoverPointIndex;
      this.update();
      return;
    }
    if (this.lastKey != "Enter") {
      this.points.push(this.localToImage(this.lastMouse));
    }
    this.updateZoom();
    if (!this.isDrawingPolygon) {
      this.points.pop();
      this.isDrawingPolygon = true;
    }

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
        if (this.lastKey == "Enter") {
          this.lastKey = null;
          this.props.submitCallback(this.points);
        }
        break;
      case "=":
        this.zoomPixels -= 10;
        this.updateZoom();
        break;
      case "-":
        this.zoomPixels += 20;
        this.updateZoom();
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
    this.ctx.fillRect(
      pt.x - this.pointSize / 4,
      pt.y - this.pointSize / 4,
      this.pointSize / 2,
      this.pointSize / 2
    );
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
      this.canvas.width / this.img.videoWidth,
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

  calculateBbox() {
    let smallestX = +Infinity;
    let smallestY = +Infinity;
    let biggestX = -Infinity;
    let biggestY = -Infinity;
    this.points.forEach((pt) => {
      smallestX = Math.min(pt.x, smallestX);
      smallestY = Math.min(pt.y, smallestY);
      biggestX = Math.max(pt.x, biggestX);
      biggestY = Math.max(pt.y, biggestY);
    });

    return [smallestX, smallestY, biggestX - smallestX, biggestY - smallestY];
  }
}

export default PolygonTool;
