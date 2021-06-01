import React from "react";

import { Container } from "react-bootstrap";

import NavbarComponent from "./components/Navbar";
import MobileHomePane from "./components/MobileHomePane";
import MobileNavigationPane from "./components/MobileNavigationPane";
class MobileMainPane extends React.Component {
  constructor(props) {
    let width = window.innerWidth;
    super(props);
    this.state = {
      screen: "home",
      imageWidth: width / 2 - 25,
    };
  }

  paneHandler(pane) {
    this.setState({
      screen: pane,
    });
    console.log(this.state);
  }

  render() {
    let displayPane;
    if (this.state.screen === "home") {
      displayPane = <MobileHomePane imageWidth={this.state.imageWidth} />;
    } else {
      displayPane = <MobileNavigationPane imageWidth={this.state.imageWidth} />;
    }
    return (
      <Container fluid>
        {displayPane}
        <NavbarComponent paneHandler={this.paneHandler.bind(this)} />
      </Container>
    );
  }
}

export default MobileMainPane;
