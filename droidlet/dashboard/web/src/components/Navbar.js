import React from "react";

import { Navbar, Nav } from "react-bootstrap";
// import './Navbar.css'
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
          <button onClick={() => this.props.paneHandler("home")}> Home </button>
          <button onClick={() => this.props.paneHandler("navigation")}>
            {" "}
            Navigation{" "}
          </button>
          <button onClick={() => this.props.paneHandler("settings")}>
            {" "}
            Settings{" "}
          </button>
        </Nav>
      </Navbar>
    );
  }
}

export default NavbarComponent;
