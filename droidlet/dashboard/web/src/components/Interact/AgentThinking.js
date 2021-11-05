/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * AgentThinking.js displays the animation that shows in between when a command is issued
   and when a response is received from the back end
 */

import React, { Component } from "react";
import Button from "@material-ui/core/Button";

import "./AgentThinking.css";

class AgentThinking extends Component {
  allowedStates = [
    "sent",
    "received",
    "thinking",
    "done_thinking",
    "executing",
  ];
  constructor(props) {
    super(props);
    this.initialState = {
      ellipsis: "",
      ellipsisInterval: null,
      commandState: "sent",
      now: null,
    };
    this.state = this.initialState;

    this.taskStackPoll = this.taskStackPoll.bind(this);
    this.issueStopCommand = this.issueStopCommand.bind(this);
    this.elementRef = React.createRef();
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  taskStackPoll(res) {
    // If we get a response of any kind, reset the timeout clock
    if (res) {
      this.setState({
        now: Date.now(),
      });
    }
    if (!res.task) {
      // If there's no task, leave this pane and go to error labeling
      this.props.goToQuestion(0);
    }
  }

  componentDidMount() {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(
      JSON.stringify({ msg: "goToAgentThinking" }),
      "*"
    );

    if (this.props.stateManager) {
      this.props.stateManager.socket.on(
        "taskStackPollResponse",
        this.taskStackPoll
      );
    }

    // General purpose interval function
    const intervalId = setInterval(() => {
      const commandState = this.props.stateManager.memory.commandState;
      console.log("Command State: " + commandState);
      this.safetyCheck(); // Check that we're in an allowed state and haven't timed out

      // Once we've added the first task, poll the agent to see when the task stack is empty
      if (commandState === "done_thinking" || commandState === "executing") {
        this.props.stateManager.socket.emit("taskStackPoll");
      }

      // Ellipsis animation and update command status
      this.setState((prevState) => {
        if (prevState.ellipsis.length > 6) {
          return {
            ellipsis: "",
            commandState: commandState,
          };
        } else {
          return {
            ellipsis: prevState.ellipsis + ".",
            commandState: commandState,
          };
        }
      });
    }, this.props.stateManager.memory.commandPollTime); // Parameterize in stateManager

    this.setState({
      ellipsisInterval: intervalId,
      commandState: this.props.stateManager.memory.commandState,
      now: Date.now(),
    });
  }

  componentWillUnmount() {
    clearInterval(this.state.ellipsisInterval);
  }

  safetyCheck() {
    // If we've gotten here during idle somehow, or timed out, escape to safety
    if (
      !this.allowedStates.includes(this.state.commandState) ||
      Date.now() - this.state.now > 40000
    ) {
      console.log("Safety check failed, exiting to Message pane.");
      this.props.goToMessage();
    }
  }

  issueStopCommand() {
    console.log("Stop command issued");
    const chatmsg = "stop";
    //add to chat history box of parent
    this.props.setInteractState({ msg: chatmsg, failed: false });
    //log message to flask
    this.props.stateManager.logInteractiondata("text command", chatmsg);
    //socket connection
    this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
    //update StateManager command state
    this.props.stateManager.memory.commandState = "sent";
  }

  renderPerformingTask() {
    return (
      <div>
        <h2>Assistant is doing the task</h2>
        <Button
          variant="contained"
          color="primary"
          onClick={this.issueStopCommand.bind(this)}
        >
          Stop
        </Button>
      </div>
    );
  }

  render() {
    return (
      <div className="container">
        {this.state.commandState === "sent" ? (
          <h2>Sending command{this.state.ellipsis}</h2>
        ) : null}
        {this.state.commandState === "received" ? (
          <h2>Command received</h2>
        ) : null}
        {this.state.commandState === "thinking" ? (
          <h2>Assistant is thinking{this.state.ellipsis}</h2>
        ) : null}
        {this.state.commandState === "done_thinking"
          ? this.renderPerformingTask()
          : null}
        {this.state.commandState === "executing"
          ? this.renderPerformingTask()
          : null}
      </div>
    );
  }
}

export default AgentThinking;
