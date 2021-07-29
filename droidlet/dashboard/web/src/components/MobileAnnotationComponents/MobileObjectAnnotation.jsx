import React from "react";

import ImageDrawer from "./ImageDrawer";
class MobileObjectAnnotation extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      currentMode: "select", // either select or annotating
    };
    this.image = this.props.image;
    this.stateManager = this.props.stateManager;
  }

  setMode(mode) {
    this.setState({
      currentMode: mode,
    });
  }

  render() {
    if (this.state.currentMode === "select") {
      return (
        <div>
          <button
            onClick={() => {
              this.setMode("annotating");
            }}
          >
            Start Annotating
          </button>
          <img
            width={this.props.imageWidth}
            height={this.props.imageWidth}
            src={this.image.src}
          />
        </div>
      );
    } else {
      return (
        <div>
          <ImageDrawer
            img={this.image}
            imageWidth={this.props.imageWidth}
            setMode={this.setMode.bind(this)}
          />
        </div>
      );
    }
  }
}

export default MobileObjectAnnotation;
