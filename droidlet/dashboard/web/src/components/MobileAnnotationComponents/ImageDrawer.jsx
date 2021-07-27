import React from "react";

class ImageDrawer extends React.Component {
  constructor(props) {
    super(props);
    this.canvasRef = React.createRef();
    this.imgRef = React.createRef();
    this.isDrawing = false;
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
    let prevMouse;
    let touch = event.touches[0];
    console.log("x: " + touch.pageX);
    console.log("y: " + touch.pageY);
    if (this.currentMouse) {
      prevMouse = this.currentMouse;
      this.currentMouse = { x: touch.pageX, y: touch.pageY };
    } else {
      this.currentMouse = { x: touch.pageX, y: touch.pageY };
    }

    // console.log('prevMouse');
    // console.log(prevMouse)
    // console.log(prevMouse.x);
    // console.log(prevMouse.y);

    if (prevMouse) {
      // draw line
      const canvas = this.canvas;
      const context = canvas.getContext("2d");

      if (context) {
        context.strokeStyle = "red";
        context.lineJoin = "round";
        context.linewidth = 5;

        context.beginPath();
        context.moveTo(prevMouse.x, prevMouse.y);
        context.lineTo(this.currentMouse.x, this.currentMouse.y);
        context.closePath();
        context.stroke();
      }
    }
  };

  render() {
    let imageSize = "500px";
    if (this.props.imageWidth) {
      imageSize = this.props.imageWidth;
    }
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
      </div>
    );
  }
}

export default ImageDrawer;
