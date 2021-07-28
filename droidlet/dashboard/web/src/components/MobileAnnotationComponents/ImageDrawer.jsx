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
    this.pointMap = [];
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
    console.log("this.scale");
    console.log(this.scale);
    this.canvas.height = this.props.img.height * this.baseScale;
    this.currentMouse = null;
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

  onTouchMove = (event) => {
    console.log("touch move");
    const canvas = this.canvas;
    console.log("this.canvas.height: " + this.canvas.height);
    console.log("this.canvas.width: " + this.canvas.width);
    let touch = event.touches[0];
    let x = (touch.clientX - canvas.offsetLeft) / this.scale;
    let y = (touch.clientY - canvas.offsetTop) / this.scale;
    console.log("x: " + x);
    console.log("y: " + y);
    if (this.currentMouse) {
      this.prevMouse = this.currentMouse;
      this.currentMouse = { x: x, y: y };
    } else {
      this.currentMouse = { x: x, y: y };
    }
    this.pointMap.push(this.currentMouse);

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
        console.log("drawing line to");
        console.log("x: " + this.currentMouse.x);
        console.log("y: " + this.currentMouse.y);
        this.ctx.closePath();
        this.ctx.stroke();
      }
    }
  };

  dataEntrySubmit(objectData) {
    this.drawing_data = objectData;
    this.annotationTags = this.drawing_data.tags;
    this.annotationName = this.drawing_data.name;

    console.log("drawing data is");
    console.log(this.drawing_data);
    this.props.setMode("select");
  }

  render() {
    let imageSize = "500px";
    if (this.props.imageWidth) {
      imageSize = this.props.imageWidth;
    }
    console.log("image size is: ");
    console.log(imageSize);

    let dataEntryX = this.canvas && this.canvas.getBoundingClientRect().left;
    let dataEntryY =
      this.canvas && this.canvas.getBoundingClientRect().bottom + 50;

    return (
      <div>
        <div>image drawer </div>

        <canvas
          ref={this.canvasRef}
          width={imageSize}
          height={imageSize}
          tabIndex="0"
          onTouchMove={this.onTouchMove}
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
