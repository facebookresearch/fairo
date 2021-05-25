import React from "react";
import ReactDOM from "react-dom";

import "bootstrap/dist/css/bootstrap.css";

import { Container, Row, Col } from "react-bootstrap";

import LiveImage from "./components/LiveImage";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import InteractApp from "./components/Interact/InteractApp";
import stateManager from "./StateManager";

console.log("new entry point established");
ReactDOM.render(
  <Container fluid>
    <Row>
      <Col>Video Feed 1</Col>
      <Col>Memory 2d </Col>
    </Row>
    <Row>
      <Col>Vidoe Feed 2</Col>
      <Col>Video Feed 3</Col>
    </Row>
    <Row>
      <Col>Nav Bar</Col>
    </Row>
  </Container>,
  document.getElementById("root")
);
