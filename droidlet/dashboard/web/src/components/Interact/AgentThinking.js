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
    this.state = {
      ellipsis: "",
      ellipsisInterval: null,
      commandState: "idle",
      now: null,
    };

    this.sendTaskStackPoll = this.sendTaskStackPoll.bind(this);
    this.receiveTaskStackPoll = this.receiveTaskStackPoll.bind(this);
    this.issueStopCommand = this.issueStopCommand.bind(this);
    this.elementRef = React.createRef();
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  sendTaskStackPoll() {
    console.log("Sending task stack poll");
    this.props.stateManager.socket.emit("taskStackPoll");
  }

  receiveTaskStackPoll(res) {
    var response = JSON.stringify(res);
    console.log("Received task stack poll response:" + response);
    // If we get a response of any kind, reset the timeout clock
    if (res) {
      this.setState({
        now: Date.now(),
      });
      if (!res.task) {
        console.log("no task on stack");
        // If there's no task, leave this pane and go to error labeling
        this.props.goToQuestion(this.props.chats.length-1);
      } else {
        // Otherwise send out a new task stack poll after a delay
        setTimeout(() => {
          this.sendTaskStackPoll();
        }, 1000);
      }
    }
  }

  componentDidMount() {
    this.props.stateManager.connect(this);  // Add to refs

    if (this.props.stateManager) {
      this.props.stateManager.socket.on( "taskStackPollResponse", this.receiveTaskStackPoll );
    }

    // Ellipsis animation and update command status
    let intervalId = setInterval(() => {
      let commandState = null;
      if (this.props.stateManager){
        commandState = this.props.stateManager.memory.commandState;
        console.log("Command State from agent thinking: " + commandState);
      }
      
      // Check that we're in an allowed state and haven't timed out
      if (this.safetyCheck()) {
        this.setState((prevState) => {
          if (prevState.commandState !== commandState) {
            // Log changes in command state to mephisto for analytics
            window.parent.postMessage(JSON.stringify({ msg: commandState }), "*");
          }
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
      } 
    }, this.props.stateManager.memory.commandPollTime);

    this.setState({
      ellipsisInterval: intervalId,
      commandState: this.props.stateManager.memory.commandState,
      now: Date.now(),
    });
  }

  componentWillUnmount() {
    clearInterval(this.state.ellipsisInterval);
    if (this.props.stateManager) {
      this.props.stateManager.disconnect(this);
      this.props.stateManager.socket.off("taskStackPollResponse", this.receiveTaskStackPoll);
    }
  }

  safetyCheck() {
    // If we've gotten here during idle somehow, or timed out, escape to safety
    if (
      !this.allowedStates.includes(this.state.commandState) ||
      (Date.now() - this.state.now) > 50000
    ) {
      console.log("Safety check failed, exiting to Message pane.");
      this.props.goToMessage();
    }
    else return true;
  }

  issueStopCommand() {
    console.log("Stop command issued");
    const chatmsg = "stop";
    //add to chat history box of parent
    this.props.setInteractState({ msg: chatmsg, timestamp: Date.now() });
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
