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
  }

  screenshot() {
    var screenshot = this.refs.webcam.getScreenshot();
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
        facingMode: { exact: "environment" },
      };

      return (
        <div>
          <Webcam
            height={this.props.imageWidth}
            width={this.props.imageWidth}
            videoConstraints={videoConstraints}
            ref="webcam"
          />
          <button onClick={this.screenshot.bind(this)}> Capture </button>
        </div>
      );
    }
    if (this.state.currentMode === "annotation") {
      return (
        <ObjectFixup
          imageWidth={this.state.screenWidth - 25}
          image={this.state.img}
          stateManager={stateManager}
          isMobile={true}
        />
      );
    }
  }
}

export default MobileCameraPane;
