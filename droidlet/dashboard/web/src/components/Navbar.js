import React from "react";

import { Navbar, Nav, NavItem } from "react-bootstrap";

class NavbarComponent extends React.Component {
  constructor(props) {
    console.log("props");
    console.log(props);
    super(props);
    this.state = props.state;
  }

  render() {
    return (
      <Navbar>
        <Nav>
          <NavItem onClick={() => this.props.paneHandler("home")}>
            {" "}
            Home
          </NavItem>
          <NavItem onClick={() => this.props.paneHandler("navigation")}>
            {" "}
            Navigation
          </NavItem>
        </Nav>
      </Navbar>
    );
  }
}

export default NavbarComponent;
