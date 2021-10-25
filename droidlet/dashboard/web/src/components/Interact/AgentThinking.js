/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * AgentThinking.js displays the animation that shows in between when a command is issued
   and when a response is received from the back end
 */

import React, { Component } from "react";

import "./AgentThinking.css";

const recognition = new window.webkitSpeechRecognition();
recognition.lang = "en-US";

class AgentThinking extends Component {
  constructor(props) {
    super(props);
    this.state = {
    };

    this.elementRef = React.createRef();
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  componentDidMount() {
    return;
  }

  componentWillUnmount() {
    return;
  }

  render() {
    return (
      <div className="thinking">
        <h3>Agent is thinking...</h3>
      </div>
    );
  }
}

export default AgentThinking;
