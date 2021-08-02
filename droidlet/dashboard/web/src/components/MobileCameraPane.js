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
      webCamPermissions: "denied",
      videoConstraints: {
        facingMode: { exact: "environment" },
        height: this.props.imageWidth,
        width: this.props.imageWidth,
      },
    };
    this.webcamRef = React.createRef();
    if (navigator.permissions && navigator.permissions.query) {
      // has side effect that mobile devices/browsers that do not support navigator.permissions wont have this check
      navigator.permissions.query({ name: "camera" }).then((permission) => {
        console.log(permission);
        this.setState({
          webCamPermissions: permission.state,
        });
      });
    }
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
      console.log("rendering");
      console.log(this.state.webCamPermissions);
      if (this.state.webCamPermissions === "denied") {
        console.log("in the if statement");
        return <div> Please grant camera permissions </div>;
      } else {
        return (
          <div>
            <Webcam
              height={this.props.imageWidth}
              width={this.props.imageWidth}
              videoConstraints={this.state.videoConstraints}
              ref={this.webcamRef}
            />
            <button onClick={this.screenshot.bind(this)}> Capture </button>
            <button onClick={this.switchCamera.bind(this)}>
              Switch Camera
            </button>
          </div>
        );
      }
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
