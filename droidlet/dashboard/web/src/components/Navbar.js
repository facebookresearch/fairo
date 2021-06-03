import React from "react";

import { Navbar, Nav } from "react-bootstrap";

class NavbarComponent extends React.Component {
  constructor(props) {
    super(props);
    this.state = props.state;
  }

  render() {
    return (
      <Navbar>
        <Nav>
          <button onClick={() => this.props.paneHandler("home")}> Home</button>
          <button onClick={() => this.props.paneHandler("navigation")}>
            {" "}
            Navigation
          </button>
        </Nav>
      </Navbar>
    );
  }
}

export default NavbarComponent;
