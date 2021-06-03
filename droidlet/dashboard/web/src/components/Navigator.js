/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import stateManager from "../StateManager";

/**
 * Currently just a keybinding wrapper that hooks
 * into the state manager and sends events upstream.
 * Later, has to be expanded to visually acknowledge
 * the keys pressed, along with their successful receipt by
 * the backend.
 */
class Navigator extends React.Component {
  constructor(props) {
    super(props);

    this.handleSubmit = this.handleSubmit.bind(this);
    this.navRef = React.createRef();
  }

  handleSubmit(event) {
    stateManager.setUrl(this.state.url);
    event.preventDefault();
  }

  componentDidMount() {
    var map = {};
    var onkey = function (e) {
      map[e.keyCode] = true;
    };
    document.addEventListener("keyup", onkey);
    setInterval(stateManager.keyHandler, 33.33, map);
  }

  render() {
    return (
      <div ref={this.navRef}>
        <p> Use the arrow keys to move the robot around</p>
      </div>
    );
  }
}

export default Navigator;
