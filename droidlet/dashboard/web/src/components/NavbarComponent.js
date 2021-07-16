/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

import { Navbar, Nav } from "react-bootstrap";
import "./Navbar.css";

class NavbarComponent extends React.Component {
  constructor(props) {
    super(props);
    this.state = props.state;
  }

  render() {
    return (
      <Navbar className="Navbar">
        <Nav>
          <button
            className="button"
            onClick={() => this.props.paneHandler("home")}
          >
            Home
          </button>
          <button
            className="button"
            onClick={() => this.props.paneHandler("navigation")}
          >
            Navigation
          </button>
          <button
            className="button"
            onClick={() => this.props.paneHandler("settings")}
          >
            Settings
          </button>
          <button
            className="button"
            onClick={() => this.props.paneHandler("annotation")}
          >
            Annotation
          </button>
          <button
            className="button"
            onClick={() => this.props.paneHandler("camera")}
          >
            Camera
          </button>
        </Nav>
      </Navbar>
    );
  }
}

export default NavbarComponent;
