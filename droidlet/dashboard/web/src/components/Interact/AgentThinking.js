/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * AgentThinking.js displays the animation that shows in between when a command is issued
   and when a response is received from the back end
 */

import React, { Component } from "react";

import "./AgentThinking.css";

class AgentThinking extends Component {
  constructor(props) {
    super(props);
    this.initialState = {
      ellipsis: "",
      ellipsisInterval: null,
    };
    this.state = this.initialState;

    this.elementRef = React.createRef();
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  componentDidMount() {
    window.parent.postMessage(JSON.stringify({ msg: "goToAgentThinking" }), "*");
    const intervalId = setInterval(() => {
      this.setState((prevState) => {
        if (prevState.ellipsis.length > 6) {
          return {
            ellipsis: "",
          };
        } else {
          return {
            ellipsis: prevState.ellipsis + ".",
          };
        }
      });
    }, 500);

    this.setState({
      ellipsisInterval: intervalId,
    });
  }

  componentWillUnmount() {
    clearInterval(this.state.ellipsisInterval);
  }

  render() {
    return (
      <div className="thinking">
        <br />
        <br />
        <br />
        <br />
        <br />
        <h2>Assistant is thinking{this.state.ellipsis}</h2>
        <br />
        <br />
        <br />
        <br />
        <br />
      </div>
    );
  }
}

export default AgentThinking;
