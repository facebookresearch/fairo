/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import "./video.css";
import { VideoMask } from "./videomask";
import PolygonTool from "./AnnotationComps/PolygonTool-Video";
import ObjectCard from "./ObjectCard";

/* React Component for video annotation */

class VideoInput extends React.Component {
  constructor(props) {
    super(props);

    this.update = this.update.bind(this);
    this.canvasClick = this.canvasClick.bind(this);

    this.currentObjId = null;
    this.newObjIndex = 0;
    this.objects = [];

    this.canvasRef = React.createRef();

    this.state = {
      mode: "select",
    };
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");

    let fileInput = document.getElementById("video-input");
    fileInput.onchange = (e) => {
      this.video = document.createElement("video");
      this.video.addEventListener("loadedmetadata", (event) => {
        this.loaded = true;

        this.baseScale = Math.min(
          this.canvas.width / this.video.videoWidth,
          (this.canvas.height - 20) / this.video.videoHeight
        );
        this.forceUpdate();
      });
      this.video.src = URL.createObjectURL(fileInput.files[0]);
      this.video.style = "width: 500px; visibility: hidden;";
      this.video.controls = false;
      this.video.loop = true;
      this.video.play();
      document.getElementById("video-container").appendChild(this.video);
    };

    setInterval(this.update, 1000 / 60);
  }

  componentDidUpdate() {
    if (this.state.mode === "select") {
      this.canvas = this.canvas = this.canvasRef.current;
      this.ctx = this.canvas.getContext("2d");
    }
  }

  update() {
    if (this.loaded && this.state.mode == "select") {
      this.ctx.resetTransform();
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.setTransform(this.baseScale, 0, 0, this.baseScale, 0, 0);
      //draw video frame
      this.ctx.drawImage(this.video, 0, 0);
      //draw points
      let pts = this.objects[this.currentObjId]?.mask.getPoints(
        this.video.currentTime
      );
      if (pts) {
        this.drawRegion(pts, "rgba(255,0,0,.5)");
      }
      //draw scrollbar
      this.ctx.resetTransform();
      this.ctx.fillStyle = "rgb(200,200,200)";
      this.ctx.fillRect(
        0,
        this.video.videoHeight * this.baseScale,
        this.canvas.width,
        20
      );
      //draw annotation windows

      //draw time marker
      this.ctx.fillStyle = "rgb(100,100,255)";
      this.ctx.fillRect(
        this.canvas.width * (this.video.currentTime / this.video.duration) -
          2.5,
        this.video.videoHeight * this.baseScale,
        5,
        20
      );
    }
  }

  render() {
    if (this.state.mode === "select") {
      return (
        <div id="workspace">
          <input type="file" id="video-input" />
          <br></br>
          <canvas
            id="display"
            ref={this.canvasRef}
            width="600px"
            height="600px"
            onClick={this.canvasClick}
          ></canvas>
          <div className="workspace-controls">
            <button
              onClick={() => {
                this.currentObjId = this.newObjIndex;
                this.newObjIndex += 1;
                this.video.pause();
                this.objects.push({
                  mask: new VideoMask(this.video),
                  id: this.currentObjId,
                  name: "new object",
                  props: [],
                });
                this.setState({
                  mode: "draw",
                });
              }}
            >
              New Object
            </button>
            <input
              type="checkbox"
              id="force-flow-box"
              defaultChecked={true}
            ></input>
            <label>Force Flow</label>
            <button
              onClick={() => {
                this.video.pause();
                if (document.getElementById("force-flow-box").checked) {
                  this.objects[this.currentObjId].mask.flowUntilCount(
                    this.video.currentTime,
                    10
                  );
                  return;
                }
                this.objects[this.currentObjId].mask.flowUntilFailure(
                  this.video.currentTime
                );
              }}
            >
              Flow Forward
            </button>
            <button
              onClick={() => {
                this.video.pause();
                if (document.getElementById("force-flow-box").checked) {
                  this.objects[this.currentObjId].mask.flowUntilCount(
                    this.video.currentTime,
                    10,
                    -1
                  );
                  return;
                }
                this.objects[this.currentObjId].mask.flowUntilFailure(
                  this.video.currentTime,
                  -1
                );
              }}
            >
              Flow Backward
            </button>
            <div className="workspace-object-list">
              {this.objects.map((obj) => {
                return (
                  <div>
                    <ObjectCard
                      name={obj.name}
                      properties={obj.props.join(", ")}
                      id={obj.id}
                      clickCallback={(e) => {
                        this.currentObjId = obj.id;
                        this.forceUpdate();
                      }}
                      editCallback={() => {
                        this.currentObjId = obj.id;
                        this.video.pause();
                        this.setState({
                          mode: "draw",
                        });
                      }}
                      updateCallback={(name, props) => {
                        this.objects[obj.id].name = name;
                        this.objects[obj.id].props = props;
                        this.forceUpdate();
                      }}
                      selected={obj.id === this.currentObjId}
                    ></ObjectCard>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      );
    } else {
      return (
        <div id="workspace">
          <input type="file" id="video-input" />
          <br></br>
          <PolygonTool
            img={this.video}
            object={"objects"}
            startPoints={this.objects[this.currentObjId].mask.getPoints(
              this.video.currentTime
            )}
            submitCallback={(pts) => {
              this.objects[this.currentObjId].mask.addGroundTruthPoints(
                pts,
                this.video.currentTime
              );
              this.setState({
                mode: "select",
              });
            }}
          ></PolygonTool>
          <div class="workspace-controls">
            <button></button>
          </div>
          <br></br>
        </div>
      );
    }
  }

  canvasClick(event) {
    if (!this.video) return;

    let rect = this.canvas.getBoundingClientRect();
    let x = event.clientX - rect.left;
    let y = event.clientY - rect.top;
    console.log(y);

    if (
      y >= this.video.videoHeight * this.baseScale &&
      y <= this.video.videoHeight * this.baseScale + 20
    ) {
      this.video.currentTime = this.video.duration * (x / this.canvas.width);
      this.video.pause();
      return;
    }

    if (this.video.paused) {
      this.video.play();
    } else {
      this.video.pause();
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

export default VideoInput;
