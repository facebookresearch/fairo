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
    this.pointSet = new Set(); // set of all the shaded pixels
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
          let radius = 2;
          this.ctx.beginPath();
          this.ctx.arc(
            this.currentMouse.x,
            this.currentMouse.y,
            radius,
            0,
            2 * Math.PI
          );
          this.addPointsWithinCircle(
            this.currentMouse.x,
            this.currentMouse.y,
            radius
          );
          this.ctx.fill();
        }
      }
    }
  };

  // add points to set
  addPointsWithinCircle(centerX, centerY, radius) {
    for (let x = centerX - radius; x < centerX + radius; x++) {
      for (let y = centerY - radius; y < centerY + radius; y++) {
        let dx = x - centerX;
        let dy = y - centerY;
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared <= radius * radius) {
          this.pointSet.add({ x: x, y: y });
        }
      }
    }
  }

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
    this.pointSet.clear();
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
