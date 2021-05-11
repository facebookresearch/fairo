/* 
Copyright (c) Facebook, Inc. and its affiliates.

Specifies a react component that takes in an imgae url, 
and provides a annotation UI for defining object relationships, similar to the 
ideas expressed in this paper: https://visualgenome.org/static/paper/Visual_Genome.pdf

props:
imgUrl: the image to annotate
*/

import React from "react";
import turk from "../turk";

class RelationshipAnnotator extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      phase: "loading",
      currentItem: 0,
    };

    this.boxes = [];
  }

  componentDidMount() {
    this.img = new Image();
    this.img.onload = () => {
      this.setState({
        phase: "writing",
        strings: [],
      });
    };
    this.img.src = this.props.imgUrl;
  }

  render() {
    if (this.state.phase === "loading") {
      return "Loading Image...";
    } else if (this.state.phase === "writing") {
      return (
        <div>
          <img src={this.img.src} width="600px"></img>
          <br></br>
          <input
            id="description-box"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                this.addString();
              }
            }}
            style={{ width: "500px" }}
            placeholder="Describe the image"
          ></input>
          <button onClick={this.addString.bind(this)}>Add (Enter)</button>
          <button onClick={this.examplesOpened.bind(this)}>
            Examples and Tips
          </button>
          <button onClick={this.writingFinished.bind(this)}>Finished</button>
          {this.state.strings.map((str, i) => {
            return (
              <div>
                <span>
                  {str}
                  <button
                    onClick={() => {
                      this.removeString(i);
                    }}
                  >
                    X
                  </button>
                </span>
              </div>
            );
          })}
        </div>
      );
    } else if (this.state.phase === "drawing") {
      return (
        <div>
          <DrawBox
            prompt={
              'Draw a box around "' +
              this.state.strings[this.state.currentItem] +
              '"'
            }
            img={this.img}
            submitCallback={this.boxFinished.bind(this)}
          />
        </div>
      );
    } else {
      return (
        <div>
          <h1>Examples</h1>
          <p>Blue Sandals</p>
          <p>A young woman wearing glasses</p>
          <p>Man taking a picture</p>
          <p>People watching birds eating</p>
          <p>Leaves on the ground</p>
          <p>A horse with a saddle on it's back</p>
          <p>A woman jogging</p>
          <p>An apple on a counter</p>
          <h1>How to get your task approved</h1>
          <p>
            We're looking for lots of different and diverse sentances to
            describe the image. Be creative! Try and get the boxes on step 2
            really close to the boundaries of what you're describing.
          </p>
          <button onClick={this.examplesClosed.bind(this)}>Back</button>
        </div>
      );
    }
  }

  addString() {
    let textBox = document.getElementById("description-box");
    let text = textBox.value.toLowerCase();
    if (text.trim() === "") {
      alert("Please write a sentance");
      return;
    }
    textBox.value = "";

    this.setState({
      strings: this.state.strings.concat(text),
    });
  }

  removeString(i) {
    var array = [...this.state.strings];
    array.splice(i, 1);
    this.setState({
      strings: array,
    });
  }

  writingFinished() {
    // TODO add checks for length and coverage
    if (this.state.strings.length < 5) {
      alert("Please write more or your task will be rejected");
      return;
    }
    this.setState({ phase: "drawing" });
  }

  examplesOpened() {
    this.setState({ phase: "examples" });
  }

  examplesClosed() {
    this.setState({ phase: "writing" });
  }

  boxFinished(box) {
    this.boxes.push(box);
    this.setState(
      {
        currentItem: (this.state.currentItem += 1),
      },
      () => {
        if (this.state.currentItem > this.state.strings.length - 1) {
          turk.submit({
            boxes: this.boxes,
            sentances: this.state.strings,
          });
        }
      }
    );
  }
}

class DrawBox extends React.Component {
  constructor(props) {
    super(props);

    this.canvasRef = React.createRef();
  }

  componentDidMount() {
    this.canvas = this.canvasRef.current;
    this.ctx = this.canvas.getContext("2d");
    this.scale = this.canvas.width / this.props.img.width;
    this.canvas.height = this.scale * this.props.img.height;
    this.box = null;

    this.update();
  }

  componentDidUpdate() {
    this.update();
  }

  render() {
    let message = this.props.prompt;
    if (this.bbox) {
      message = "Drag the green handels to adjust";
    }

    return (
      <div>
        <p>{message}</p>
        <canvas
          ref={this.canvasRef}
          width="600px"
          height="600px"
          onMouseDown={this.mouseDown.bind(this)}
          onMouseUp={this.mouseUp.bind(this)}
          onMouseMove={this.mouseMove.bind(this)}
          onMouseLeave={this.mouseUp.bind(this)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              if (this.bbox) {
                this.props.submitCallback(this.bbox);
                this.bbox = null;
                this.update();
              } else {
                alert(this.props.prompt);
              }
            }
          }}
          tabIndex={0}
        ></canvas>
        <br></br>
        <button
          onClick={() => {
            this.bbox = null;
            this.update();
          }}
        >
          Clear
        </button>
        <button
          onClick={() => {
            if (this.bbox) {
              this.props.submitCallback(this.bbox);
              this.bbox = null;
              this.update();
            } else {
              alert(this.props.prompt);
            }
          }}
        >
          Submit
        </button>
      </div>
    );
  }

  update() {
    //clear and transform
    this.ctx.resetTransform();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
    //Draw image scaled and repostioned
    this.ctx.drawImage(this.props.img, 0, 0);
    //Draw box
    if (this.bbox) {
      this.ctx.lineWidth = 5;
      this.ctx.strokeRect(
        this.bbox.pt1.x,
        this.bbox.pt1.y,
        this.bbox.pt2.x - this.bbox.pt1.x,
        this.bbox.pt2.y - this.bbox.pt1.y
      );

      this.hoverName = null;
      this.drawPoint(this.bbox.pt1, "pt1");
      this.drawPoint(this.bbox.pt2, "pt2");
    }
  }

  drawPoint(pt, name = null) {
    this.ctx.fillStyle = "green";
    if (this.distance(pt, this.lastMouse) < 30) {
      this.ctx.fillStyle = "orange";
      if (name) {
        this.hoverName = name;
      }
    }
    this.ctx.fillRect(pt.x - 15, pt.y - 15, 30, 30);
  }

  distance(pt1, pt2) {
    return Math.max(Math.abs(pt1.x - pt2.x), Math.abs(pt1.y - pt2.y));
  }

  setMouse(e) {
    var rect = this.canvas.getBoundingClientRect();
    this.lastMouse = {
      x: (e.clientX - rect.left) / this.scale,
      y: (e.clientY - rect.top + 1) / this.scale,
    };
  }

  mouseDown(e) {
    this.setMouse(e);
    if (!this.bbox) {
      this.drawingBox = true;
      this.bbox = {
        pt1: this.lastMouse,
        pt2: this.lastMouse,
      };
    } else {
      if (this.hoverName) {
        this.draggingName = this.hoverName;
      }
    }
    this.update();
  }

  mouseUp(e) {
    this.setMouse(e);
    this.draggingName = null;
    this.drawingBox = false;
    this.update();
  }

  mouseMove(e) {
    if (this.drawingBox) {
      this.bbox.pt2 = this.lastMouse;
    } else if (this.draggingName) {
      this.bbox[this.draggingName] = this.lastMouse;
    }
    this.setMouse(e);

    this.update();
  }
}

export default RelationshipAnnotator;
