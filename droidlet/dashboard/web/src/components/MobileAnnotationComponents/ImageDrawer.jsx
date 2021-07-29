import React from "react";

import DataEntry from "./DataEntry";

class ImageDrawer extends React.Component {
  constructor(props) {
    super(props);
    this.canvasRef = React.createRef();
    this.imgRef = React.createRef();
    this.isDrawing = false;
    this.annotationName = null;
    this.annotationTags = null;
    this.pointList = []; // list of all the points on the mask
    this.isDrawing = true;
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");
    this.Offset = {
      x: 0,
      y: 0,
    };
    this.baseScale = this.canvas.width / this.props.img.width;
    this.scale = this.baseScale;
    this.canvas.height = this.props.img.height * this.baseScale;
    this.currentMouse = null;
    this.firstCoordinate = null;
    this.prevMouse = null;
    this.update();
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
    this.ctx.drawImage(this.props.img, 0, 0);
  }

  onTouchEnd = (event) => {
    this.isDrawing = false;
    this.ctx.strokeStyle = "red";
    this.ctx.lineJoin = "round";
    this.ctx.linewidth = 5;

    this.ctx.beginPath();
    this.ctx.moveTo(this.currentMouse.x, this.currentMouse.y);
    this.ctx.lineTo(this.firstCoordinate.x, this.firstCoordinate.y);
    this.ctx.closePath();
    this.ctx.stroke();
    this.isDrawing = false;
    this.pointList.push(this.firstCoordinate);
  };

  onTouchMove = (event) => {
    if (this.isDrawing) {
      const canvas = this.canvas;
      let touch = event.touches[0];
      let x = (touch.clientX - canvas.offsetLeft) / this.scale;
      let y = (touch.clientY - canvas.offsetTop) / this.scale;
      if (this.currentMouse) {
        this.prevMouse = this.currentMouse;
        this.currentMouse = { x: x, y: y };
      } else {
        this.currentMouse = { x: x, y: y };
        this.firstCoordinate = this.currentMouse;
      }
      this.pointList.push(this.currentMouse);

      if (this.prevMouse) {
        // draw line
        this.ctx = this.canvas.getContext("2d");

        if (this.ctx) {
          this.ctx.strokeStyle = "red";
          this.ctx.lineJoin = "round";
          this.ctx.linewidth = 5;

          this.ctx.beginPath();
          this.ctx.moveTo(this.prevMouse.x, this.prevMouse.y);
          this.ctx.lineTo(this.currentMouse.x, this.currentMouse.y);
          this.ctx.closePath();
          this.ctx.stroke();
        }
      }
    }
  };

  // submit annotation to backend
  dataEntrySubmit(objectData) {
    this.drawing_data = objectData;
    this.annotationTags = this.drawing_data.tags;
    this.annotationName = this.drawing_data.name;
    // reset the setting on mobileObjectAnnotation
    this.props.setMode("select");
  }

  // clears all the lines on the screen
  clearMask() {
    this.pointList = [];
    this.update();
  }

  render() {
    let imageSize = "500px";
    if (this.props.imageWidth) {
      imageSize = this.props.imageWidth;
    }

    let dataEntryX = this.canvas && this.canvas.getBoundingClientRect().left;
    let dataEntryY =
      this.canvas && this.canvas.getBoundingClientRect().bottom + 50;

    return (
      <div>
        <div> Outline image to annotate </div>
        <button onClick={this.clearMask.bind(this)}>Clear Mask</button>
        <canvas
          style={{ touchAction: "none" }}
          ref={this.canvasRef}
          width={imageSize}
          height={imageSize}
          tabIndex="0"
          onTouchMove={this.onTouchMove}
          onTouchEnd={this.onTouchEnd}
          onMouseUp={this.mouseEvent}
        ></canvas>

        <DataEntry
          x={dataEntryX}
          y={dataEntryY}
          onSubmit={this.dataEntrySubmit.bind(this)}
          includeSubmitButton={true}
          isMobile={this.props.isMobile}
        />
      </div>
    );
  }
}

export default ImageDrawer;
