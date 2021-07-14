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
    let screenshot = this.webcamRef.current.getScreenshot();
    let asImage = new Image(this.props.imageWidth, this.props.imageWidth);
    asImage.src = screenshot;
    this.setState({
      currentMode: "annotation",
      img: asImage,
    });
  }

  componentDidMount() {
    // check video constraint after 1 second to give time for the webcam to finish rendering
    setTimeout(
      function () {
        this.checkVideoConstraint();
      }.bind(this),
      1000
    );
  }

  /**
   * check if device has environment camera
   * if not, use the device selfie camera
   * only needed for development purposes, as every mobile device should have an environment camera
   */
  checkVideoConstraint() {
    if (true) {
      if (this.webcamRef.current) {
        let screenshot = this.webcamRef.current.getScreenshot();
        if (!screenshot) {
          this.setState({
            videoConstraints: {
              facingMode: "user",
              height: this.props.imageWidth,
              width: this.props.imageWidth,
            },
          });
        }
      }
    }
  }

  render() {
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
