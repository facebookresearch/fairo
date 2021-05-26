import React from "react";
import ReactDOM from "react-dom";

import "bootstrap/dist/css/bootstrap.css";

import { Container, Row, Col } from "react-bootstrap";

import LiveImage from "./components/LiveImage";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import stateManager from "./StateManager";
import InteractApp from "./components/Interact/InteractApp";
import Memory2D from "./components/Memory2D";

console.log("new entry point established");
ReactDOM.render(
  <Container fluid>
    <Row>
      <Col>
        Video Feed 1
        <LiveImage
          type={"rgb"}
          height={320}
          width={320}
          offsetH={0}
          offsetW={0}
          stateManager={stateManager}
          isMobile={true}
        />
      </Col>
      <Col>
        Memory 2d
        {/* <Memory2D
          stateManager={stateManager}
        /> */}
      </Col>
    </Row>
    <Row>
      <Col>
        Video Feed 2
        <LiveImage
          type={"depth"}
          height={320}
          width={320}
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
          height={320}
          width={320}
          offsetH={0}
          offsetW={0}
          stateManager={stateManager}
          isMobile={true}
        />
      </Col>
    </Row>
    <Row>
      <p>Interact App </p>
      <InteractApp stateManager={stateManager} />
    </Row>

    <Row>
      <Col>Nav Bar</Col>
    </Row>
  </Container>,
  document.getElementById("root")
);
