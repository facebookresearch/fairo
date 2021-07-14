import React from "react";
import Webcam from "react-webcam";
import ObjectFixup from "./ObjectFixup";
import stateManager from ".././StateManager";

class MobileCameraPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      currentMode: "camera",
      img: null,
      videoConstraints: {
        facingMode: { exact: "environment" },
        height: this.props.imageWidth,
        width: this.props.imageWidth,
      },
    };
    this.webcamRef = React.createRef();
  }

  screenshot() {
    var screenshot = this.webcamRef.current.getScreenshot();
    console.log("screenshot");
    console.log(screenshot);
    let asImage = new Image(this.props.imageWidth, this.props.imageWidth);
    asImage.src = screenshot;
    this.setState({
      currentMode: "annotation",
      img: asImage,
    });
  }

  componentDidMount() {
    console.log("post mount");
    if (this.webcamRef.current) {
      // if device has no environment camera, use the selfie camera
      if (!this.webcamRef.current.getScreenshot()) {
        console.log("switching camera");
        this.setState({
          videoConstraints: {
            facingMode: "user",
            height: this.props.imageWidth,
            width: this.props.imageWidth,
          },
        });
        console.log("this.state");
        console.log(this.state);
      }
    }
  }

  render() {
    console.log("camera render");
    if (this.state.currentMode === "camera") {
      return (
        <div>
          <Webcam
            height={this.props.imageWidth}
            width={this.props.imageWidth}
            videoConstraints={this.state.videoConstraints}
            ref={this.webcamRef}
          />
          <button onClick={this.screenshot.bind(this)}> Capture </button>
        </div>
      );
    }
    if (this.state.currentMode === "annotation") {
      return (
        <ObjectFixup
          imageWidth={this.props.imageWidth}
          image={this.state.img}
          stateManager={stateManager}
          isMobile={true}
          isFromCamera={true}
          isFirstOpen={true}
        />
      );
    }
  }
}

export default MobileCameraPane;
