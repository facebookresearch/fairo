import React from "react";
import ReactDOM from "react-dom";

import "bootstrap/dist/css/bootstrap.css";

import { Navbar, Nav, NavItem, NavDropdown, MenuItem } from "react-bootstrap";
import { Container, Row, Col } from "react-bootstrap";

import LiveImage from "./components/LiveImage";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import stateManager from "./StateManager";
import InteractApp from "./components/Interact/InteractApp";
import Memory2D from "./components/Memory2D";

console.log("new entry point established");
let width = window.innerWidth;
let imageWidth = width / 2 - 25;
console.log("image width is:");
console.log(imageWidth);
ReactDOM.render(
  <Container fluid>
    <Row>
      <Col>
        Video Feed 1
        <LiveImage
          type={"rgb"}
          height={imageWidth}
          width={imageWidth}
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
          dimensions={imageWidth}
        />
      </Col>
    </Row>
    <Row>
      <Col>
        Video Feed 2
        <LiveImage
          type={"depth"}
          height={imageWidth}
          width={imageWidth}
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
          height={imageWidth}
          width={imageWidth}
          offsetH={0}
          offsetW={0}
          stateManager={stateManager}
          isMobile={true}
        />
      </Col>
    </Row>
    <Navbar>
      <Navbar.Header>
        <Navbar.Brand>
          <a href="#home">My Brand</a>
        </Navbar.Brand>
      </Navbar.Header>
      <Nav>
        <NavItem href="#">Home</NavItem>
        <NavItem href="#">About</NavItem>
        <NavItem href="#">FAQ</NavItem>
        <NavItem href="#">Contact Us</NavItem>
      </Nav>
    </Navbar>
    <Row>
      <p>Interact App </p>
      <InteractApp stateManager={stateManager} />
    </Row>
  </Container>,
  document.getElementById("root")
);
