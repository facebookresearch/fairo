/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";

import { Row, Col } from "react-bootstrap";

import LiveImage from "./LiveImage";
import LiveHumans from "./LiveHumans";
import stateManager from ".././StateManager";
import Memory2D from "./Memory2D";
import Settings from "./Settings";

class MobileSettingsPane extends React.Component {
  render() {
    return (
      <div>
        <Row>
          <Col>
            Video Feed 1
            <LiveImage
              type={"rgb"}
              height={this.props.imageWidth}
              width={this.props.imageWidth}
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
              dimensions={this.props.imageWidth}
            />
          </Col>
        </Row>
        <Row>
          <Col>
            Video Feed 2
            <LiveImage
              type={"depth"}
              height={this.props.imageWidth}
              width={this.props.imageWidth}
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
              height={this.props.imageWidth}
              width={this.props.imageWidth}
              offsetH={0}
              offsetW={0}
              stateManager={stateManager}
              isMobile={true}
            />
          </Col>
        </Row>
        <Settings stateManager={stateManager} isMobile={true} />
      </div>
    );
  }
}

export default MobileSettingsPane;
