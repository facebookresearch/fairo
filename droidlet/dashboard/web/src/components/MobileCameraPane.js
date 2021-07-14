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
    };
    this.webcamRef = React.createRef();
  }

  screenshot() {
    var screenshot = this.webcamRef.current.getScreenshot();
    let asImage = new Image(this.props.imageWidth, this.props.imageWidth);
    asImage.src = screenshot;
    this.setState({
      currentMode: "annotation",
      img: asImage,
    });
  }

  render() {
    if (this.state.currentMode === "camera") {
      const videoConstraints = {
        // facingMode: { exact: "environment" },
        height: this.props.imageWidth,
        width: this.props.imageWidth,
      };

      return (
        <div>
          <Webcam
            height={this.props.imageWidth}
            width={this.props.imageWidth}
            videoConstraints={videoConstraints}
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
