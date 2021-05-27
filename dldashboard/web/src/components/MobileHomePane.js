import React from "react";

import { Container, Row, Col } from "react-bootstrap";

import LiveImage from "./LiveImage";
import LiveHumans from "./LiveHumans";
import stateManager from ".././StateManager";
import InteractApp from "./Interact/InteractApp";
import Memory2D from "./Memory2D";

class MobileHomePane extends React.Component {
  constructor(props) {
    let width = window.innerWidth;
    super(props);
    console.log("the props are");
    console.log(props);
    let imageWidth = this.props.imageWidth;
  }

  render() {
    console.log("image width is");
    console.log(this.props.imageWidth);
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
        <Row>
          <InteractApp stateManager={stateManager} />
        </Row>
      </div>
    );
  }
}

export default MobileHomePane;
