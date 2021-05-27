import React from "react";

import { Container, Row, Col } from "react-bootstrap";

import LiveImage from "./components/LiveImage";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import stateManager from "./StateManager";
import InteractApp from "./components/Interact/InteractApp";
import Memory2D from "./components/Memory2D";
import NavbarComponent from "./components/Navbar";

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
    if (this.state.screen === "home") {
      return (
        <Container fluid>
          <Row>
            <Col>
              Video Feed 1
              <LiveImage
                type={"rgb"}
                height={this.state.imageWidth}
                width={this.state.imageWidth}
                offsetH={0}
                offsetW={0}
                stateManager={stateManager}
                isMobile={true}
              />
            </Col>
            <Col>
              Memory 2d
              <Memory2D
                stateManager={stateManager}
                isMobile={true}
                dimensions={this.state.imageWidth}
              />
            </Col>
          </Row>
          <Row>
            <Col>
              Video Feed 2
              <LiveImage
                type={"depth"}
                height={this.state.imageWidth}
                width={this.state.imageWidth}
                offsetH={0}
                offsetW={0}
                stateManager={stateManager}
                isMobile={true}
              />
            </Col>

            <Col>
              Video Feed 3
              <LiveHumans
                type={"depth"}
                height={this.state.imageWidth}
                width={this.state.imageWidth}
                offsetH={0}
                offsetW={0}
                stateManager={stateManager}
                isMobile={true}
              />
            </Col>
          </Row>
          <Row>
            <InteractApp stateManager={stateManager} />
          </Row>
          <NavbarComponent paneHandler={this.paneHandler.bind(this)} />
        </Container>
      );
    } else {
      return <p> hai </p>;
    }
  }
}

export default MobileMainPane;
