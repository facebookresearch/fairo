/* 
Copyright (c) Facebook, Inc. and its affiliates.

Specifies a react component that takes in a list of objects, 
their defining vertices, and colors, and renders them all in a canvas. 
Useful for showing the users what progress they've made in outlining objects, 
and for seeing data coming in from turk tasks.
*/

import React from "react";

/*Props

img (Javascript Image Object)
    Image to draw segments on

objects (List of object names to draw [keys to pointMap])

pointMap (Map from object names to arrays of points)
    Defines the borders of objects to be drawn

colors (List of colors to use to draw the segments)
    list of colors to use. If the length of the colors array is less than 
    the objects to be drawn, colors will be recycled (harder to read)

*/

class SegmentRenderer extends React.Component {
  constructor(props) {
    super(props);

    this.update = this.update.bind(this);
    this.drawLine = this.drawLine.bind(this);

    this.canvasRef = React.createRef();
    this.imgRef = React.createRef();
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
    this.update();
  }

  render() {
    return (
      <div>
        <canvas
          ref={this.canvasRef}
          width="500px"
          height="500px"
          tabIndex="0"
          onClick={this.props.onClick}
          onMouseMove={this.onMouseMove}
          onKeyDown={this.keyDown}
        ></canvas>
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
    this.ctx.drawImage(this.props.img, 0, 0);
    //Draw regions
    for (let i = 0; i < this.props.objects.length; i++) {
      let pts_arr = this.props.pointMap[this.props.objects[i]];
      if (pts_arr.length > 0) {
        let color =
          this.props.colors[i % this.props.colors.length] || "rgba(0,200,0,.5)";
        for (let i = 0; i < pts_arr.length; i++) {
          // Must denormalize points
          this.drawRegion(
            pts_arr[i].map((pt) => ({
              x: pt.x * this.canvas.width,
              y: pt.y * this.canvas.height,
            })),
            color
          );
        }
      }
    }
  }

  drawLine(pt1, pt2) {
    this.ctx.beginPath();
    this.ctx.moveTo(pt1.x, pt1.y);
    this.ctx.lineTo(pt2.x, pt2.y);
    this.ctx.stroke();
  }

  drawRegion(points, color) {
    if (points.length < 3) {
      return;
    }
    let region = new Path2D();
    region.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      region.lineTo(points[i].x, points[i].y);
    }
    region.closePath();
    this.ctx.fillStyle = color;
    this.ctx.fill(region, "evenodd");
  }
}

export default SegmentRenderer;
