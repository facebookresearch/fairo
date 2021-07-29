import React from "react";
import Webcam from "react-webcam";
import ObjectFixup from "./ObjectFixup";
import stateManager from ".././StateManager";
import MobileObjectAnnotation from "./MobileAnnotationComponents/MobileObjectAnnotation";

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
    console.log("asImage");
    console.log(asImage);
    asImage.src = screenshot;
    this.setState({
      currentMode: "annotation",
      img: asImage,
    });
  }

  switchCamera() {
    let newFacingMode = this.state.videoConstraints.facingMode;
    if (newFacingMode === "user") {
      newFacingMode = { exact: "environment" };
    } else {
      newFacingMode = "user";
    }
    this.setState({
      videoConstraints: {
        facingMode: newFacingMode,
        height: this.props.imageWidth,
        width: this.props.imageWidth,
      },
    });
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
          <button onClick={this.switchCamera.bind(this)}>Switch Camera</button>
        </div>
      );
    }
    if (this.state.currentMode === "annotation") {
      if (stateManager.useDesktopComponentOnMobile) {
        return (
          <ObjectFixup
            imageWidth={this.props.imageWidth}
            image={this.state.img}
            stateManager={stateManager}
            isMobile={true}
            isFromCamera={true}
          />
        );
      } else {
        return (
          <MobileObjectAnnotation
            imageWidth={this.props.imageWidth - 25}
            image={this.state.img}
            stateManager={stateManager}
          />
        );
      }
    }
  }
}

export default MobileCameraPane;
