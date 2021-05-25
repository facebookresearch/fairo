/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import ReactDOM from "react-dom";

import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";

import "./index.css";

const mobileLayout = () => {
  ReactDOM.render(
    <React.StrictMode>
      <Container>
        <Row>
          <Col> 1 of 1 </Col>
        </Row>
      </Container>
    </React.StrictMode>,
    document.getElementById("root")
  );
};

export default mobileLayout;
