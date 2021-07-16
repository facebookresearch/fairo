/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";

/* Props

img (Javascript Image object)
    *Loaded* image to be drawn to the canvas and traced

object (String) 
    Predicted name of the object

bbox (array of [x, y, width, height] measured from the top left)
    Bounding box on the object to be relabled

submitCallback (pts) 
    Function that handles submission. Takes an array of points 
    (outline of the object)
*/

class MaskCorrectionTool extends React.Component {
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

    this.isDrawingPolygon = false;
    this.lastMouse = {
      x: 0,
      y: 0,
    };
    this.dragging = false;

    this.canvasRef = React.createRef();

    this.zoomPixels = this.props.img.width / 2.5;
    this.pointSize = 12;
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");

    this.points = [];

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
        <p>
          Click the <b>edge</b> of the {this.props.object} to start.
        </p>
        <div>
          <button
            onClick={() => {
              if (this.points.length > 0) {
                this.points.pop();
                this.update();
              }
            }}
          >
            Undo
          </button>
          <button
            onClick={() => {
              this.props.submitCallback(this.points);
            }}
          >
            Finish
          </button>
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
    //Draw box around target
    this.ctx.beginPath();
    this.ctx.strokeStyle = "rgba(255,0,0,.5)";
    this.ctx.lineWidth = 10;
    this.ctx.rect(...this.props.bbox);
    this.ctx.stroke();
    //Draw points and lines
    this.ctx.strokeStyle = "black";
    this.ctx.lineWidth = 4;
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
    // uncomment to enable zooming (slower but more acurate data)
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
    }
    this.lastKey = e.key;
    this.update();
  }

  drawPoint(pt) {
    this.ctx.fillStyle = "green";
    if (
      this.distance(pt, this.localToImage(this.lastMouse)) <
      this.pointSize / 2
    ) {
      this.ctx.fillStyle = "yellow";
    }
    this.ctx.fillRect(
      pt.x - this.pointSize / 2,
      pt.y - this.pointSize / 2,
      this.pointSize,
      this.pointSize
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

export default MaskCorrectionTool;
