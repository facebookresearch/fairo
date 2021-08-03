/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

import { Container } from "react-bootstrap";

import NavbarComponent from "./components/NavbarComponent";
import MobileHomePane from "./components/MobileHomePane";
import MobileNavigationPane from "./components/MobileNavigationPane";
import MobileSettingsPane from "./components/MobileSettingsPane";
import ObjectFixup from "./components/ObjectFixup";
import MobileObjectAnnotation from "./components/MobileAnnotationComponents/MobileObjectAnnotation";
import MobileCameraPane from "./components/MobileCameraPane";

import stateManager from "./StateManager";
class MobileMainPane extends React.Component {
  constructor(props) {
    let width = window.innerWidth;
    super(props);
    // screen is the current pane
    // width is the size of the feeds
    // object_rgb is the object feed that gets fed into annotation pane
    this.state = {
      screen: "home",
      imageWidth: width * 0.4,
      screenWidth: width,
      objectRGB: null,
    };
  }

  componentDidMount() {
    if (stateManager) stateManager.connect(this);
  }

  paneHandler(pane) {
    this.setState({
      screen: pane,
    });
  }

  render() {
    let displayPane;
    if (this.state.screen === "home") {
      displayPane = <MobileHomePane imageWidth={this.state.imageWidth} />;
    } else if (this.state.screen === "navigation") {
      displayPane = <MobileNavigationPane imageWidth={this.state.imageWidth} />;
    } else if (this.state.screen === "settings") {
      displayPane = <MobileSettingsPane imageWidth={this.state.imageWidth} />;
    } else if (this.state.screen === "annotation") {
      if (stateManager.useDesktopComponentOnMobile) {
        displayPane = (
          <ObjectFixup
            imageWidth={this.state.screenWidth - 25}
            image={this.state.objectRGB}
            stateManager={stateManager}
            isMobile={true}
          />
        );
      } else {
        displayPane = (
          <MobileObjectAnnotation
            imageWidth={this.state.screenWidth - 25}
            image={this.state.objectRGB}
            stateManager={stateManager}
          />
        );
      }
    } else if (this.state.screen === "camera") {
      displayPane = (
        <MobileCameraPane imageWidth={this.state.screenWidth - 25} />
      );
    }
    return (
      <Container fluid>
        <div style={{ paddingBottom: 200 }}>{displayPane}</div>
        <div>
          <NavbarComponent paneHandler={this.paneHandler.bind(this)} />
        </div>
      </Container>
    );
  }
}

export default MobileMainPane;
