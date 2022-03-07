/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * InteractApp.js implements ASR, send the chat message, switch to the fail or back to settings view
 */

import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import "./InteractApp.css";

class InteractApp extends Component {
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
      recognizing: false,
      enableVoice: false,
      connected: false,
      ellipsis: "",
      commandState: "idle",
      now: null,
      disableInput: false,
      disableStopButton: true,
      currentView: 1,
      lastChatActionDict: "",
      status: "",
      chats: [{ msg: "", timestamp: Date.now() }],
      failidx: -1,
      agent_replies: [{}],
      agentType: null,
      isTurk: false,
    };
    this.state = this.initialState;
    this.elementRef = React.createRef();
    this.bindKeyPress = this.handleKeyPress.bind(this); // this is used in keypressed event handling
    this.sendTaskStackPoll = this.sendTaskStackPoll.bind(this);
    this.receiveTaskStackPoll = this.receiveTaskStackPoll.bind(this);
    this.issueStopCommand = this.issueStopCommand.bind(this);
    this.handleAgentThinking = this.handleAgentThinking.bind(this);
    this.handleClearInterval = this.handleClearInterval.bind(this);
    this.goToQuestionWindow = this.goToQuestionWindow.bind(this);
    this.MessageRef = React.createRef();
    this.intervalId = null;
    this.messagesEnd = null;
    this.addNewAgentReplies = this.addNewAgentReplies.bind(this);
  }

  setAnswerIndex(index) {
    this.setState({
      answerIndex: index,
    });
  }

  updateChat(chat) {
    // make a shallow copy of chats
    var new_chats = [...this.state.chats];
    new_chats.push(chat);
    this.setState({ chats: new_chats });
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
        // If there's no task, leave this pane
        // If it's a HIT go to error labeling, else back to Message
        if (this.state.isTurk) {
          this.goToQuestion(this.state.chats.length - 1);
        } else {
          this.handleClearInterval();
        }
      } else {
        // Otherwise send out a new task stack poll after a delay
        setTimeout(() => {
          this.sendTaskStackPoll();
        }, 1000);
      }
    }
  }

  issueStopCommand() {
    console.log("Stop command issued");
    const chatmsg = "stop";
    //add to chat history box of parent
    this.updateChat({ msg: chatmsg, timestamp: Date.now() });
    //log message to flask
    this.props.stateManager.logInteractiondata("text command", chatmsg);
    //socket connection
    this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
    //update StateManager command state
    this.props.stateManager.memory.commandState = "sent";
  }

  renderChatHistory() {
    // Pull in user chats and agent replies, filter out any empty ones
    let chats = this.state.chats.filter((chat) => chat.msg !== "");
    let replies = this.state.agent_replies.filter((reply) => reply.msg !== "");
    chats = chats.filter((chat) => chat.msg);
    replies = replies.filter((reply) => reply.msg);
    // Label each chat based on where it came from
    chats.forEach((chat) => (chat["sender"] = "message user"));
    replies.forEach((reply) => (reply["sender"] = "message agent"));
    // Strip out the 'Agent: ' prefix if it's there
    replies.forEach(function (reply) {
      if (reply["msg"].includes("Agent: ")) {
        reply["msg"] = reply["msg"].substring(7);
      }
    });
    // Zip it into one list, sort by timestamp, and send it off to be rendered
    let chat_history = chats.concat(replies);
    chat_history.sort(function (a, b) {
      return a.timestamp - b.timestamp;
    });

    return chat_history.map((chat) =>
      React.cloneElement(
        <li className="message-item" key={chat.timestamp.toString()}>
          <div className={chat.sender}>{chat.msg}</div>
          {chat.isQestion && (
            <div className="answer-buttons">
              <Button
                variant="contained"
                color="primary"
                className="yes-button"
              >
                Yes
              </Button>
              <Button variant="contained" color="primary" className="no-button">
                No
              </Button>
            </div>
          )}
        </li>
      )
    );
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  handleKeyPress(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      this.handleSubmit();
    }
  }

  componentDidMount() {
    document.addEventListener("keypress", this.bindKeyPress);
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      this.setState({
        isTurk: this.props.stateManager.memory.isTurk,
        agent_replies: this.props.stateManager.memory.agent_replies,
        connected: this.props.stateManager.connected,
      });
    }
    // Scroll messsage panel to bottom
    this.scrollToBottom();
  }

  componentWillUnmount() {
    document.removeEventListener("keypress", this.bindKeyPress);
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
  }

  getUrlParameterByName(name) {
    var match = RegExp("[?&]" + name + "=([^&]*)").exec(window.location.search);
    return match && decodeURIComponent(match[1].replace(/\+/g, " "));
  }

  goToQuestion(idx) {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(JSON.stringify({ msg: "goToQuestion" }), "*");

    // Wait for the logical form of last chat and show the Fail page
    this.props.stateManager.socket.on(
      "setLastChatActionDict",
      this.goToQuestionWindow
    );

    // Send request to retrieve the logic form of last sent command
    this.props.stateManager.socket.emit(
      "getChatActionDict",
      this.state.chats[idx]["msg"]
    );
  }

  goToQuestionWindow() {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(
      JSON.stringify({ msg: "goToQuestionWindow" }),
      "*"
    );
    // Don't proliferate sio listeners
    this.props.stateManager.socket.off(
      "setLastChatActionDict",
      this.goToQuestionWindow
    );

    const chats_len = this.state.chats.length;

    this.addNewAgentReplies(
      true,
      "Did I successfully do the task you asked me to complete?"
    );

    this.setState({
      agent_replies: this.props.stateManager.memory.agent_replies,
      currentView: 2,
      chats: this.state.chats,
      failidx: chats_len - 1,
    });
  }

  handleSubmit() {
    //get the message
    var chatmsg = document.getElementById("msg").value;
    if (chatmsg.replace(/\s/g, "") !== "") {
      //add to chat history box of parent
      this.updateChat({ msg: chatmsg, timestamp: Date.now() });
      //log message to flask
      this.props.stateManager.logInteractiondata("text command", chatmsg);
      //log message to Mephisto
      window.parent.postMessage(
        JSON.stringify({ msg: { command: chatmsg } }),
        "*"
      );
      //send message to TurkInfo
      this.props.stateManager.sendCommandToTurkInfo(chatmsg);
      //socket connection
      this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
      //update StateManager command state
      this.props.stateManager.memory.commandState = "sent";
      //clear the textbox
      document.getElementById("msg").value = "";
      //clear the agent reply that will be shown in the question pane
      this.props.stateManager.memory.last_reply = "";
      //execute agent thinking function if it makes sense
      if (this.state.agentType === "craftassist") {
        this.handleAgentThinking();
      }
    }
  }

  // Merge agent thinking functionality
  handleAgentThinking() {
    if (this.props.stateManager) {
      this.props.stateManager.socket.on(
        "taskStackPollResponse",
        this.receiveTaskStackPoll
      );
    }

    this.intervalId = setInterval(() => {
      let commandState = null;

      if (this.props.stateManager) {
        commandState = this.props.stateManager.memory.commandState;
        console.log("Command State from agent thinking: " + commandState);
      }

      // Check that we're in an allowed state and haven't timed out
      if (this.safetyCheck()) {
        this.setState((prevState) => {
          if (prevState.commandState !== commandState) {
            // Log changes in command state to mephisto for analytics
            window.parent.postMessage(
              JSON.stringify({ msg: commandState }),
              "*"
            );
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
      commandState: this.props.stateManager.memory.commandState,
      now: Date.now(),
    });
  }

  safetyCheck() {
    // If we've gotten here during idle somehow, or timed out, escape to safety
    if (
      !this.allowedStates.includes(this.state.commandState) ||
      Date.now() - this.state.now > 50000
    ) {
      console.log("Safety check failed, exiting to Message pane.");
      this.handleClearInterval();
      return false;
    } else return true;
  }

  // Stop sending command
  handleClearInterval() {
    clearInterval(this.intervalId);
    if (this.props.stateManager) {
      this.props.stateManager.socket.off(
        "taskStackPollResponse",
        this.receiveTaskStackPoll
      );
      this.setState({
        disableInput: false,
        agent_replies: this.props.stateManager.memory.agent_replies,
        disableStopButton: true,
      });
    }
  }

  // Scroll to bottom when submit new message
  scrollToBottom = () => {
    if (this.messagesEnd)
      this.messagesEnd.scrollIntoView({ behavior: "smooth" });
  };

  componentDidUpdate(prevProps, prevState) {
    // Show command message like an agent reply
    if (this.state.commandState !== prevState.commandState) {
      let command_message = "";
      let disableInput = true;
      let disableStopButton = this.state.disableStopButton;
      if (this.state.commandState === "sent") {
        command_message = "Sending command...";
        disableStopButton = true;
      } else if (this.state.commandState === "received") {
        command_message = "Command received";
        disableStopButton = true;
      } else if (this.state.commandState === "thinking") {
        command_message = "Assistant is thinking...";
        disableStopButton = true;
      } else if (this.state.commandState === "done_thinking") {
        command_message = "Assistant is doing the task";
        disableStopButton = false;
      } else if (this.state.commandState === "executing") {
        command_message = "Assistant is doing the task";
        disableStopButton = false;
      }
      if (command_message) {
        const newAgentReplies = [
          ...this.state.agent_replies,
          { msg: command_message, timestamp: Date.now() },
        ];
        this.setState({
          agent_replies: newAgentReplies,
          disableInput: disableInput,
          disableStopButton: disableStopButton,
        });
      }
    }
    // Scroll messsage panel to bottom
    this.scrollToBottom();
  }

  addNewAgentReplies(isQestion, agentReply) {
    console.log(agentReply);
    const newAgentReplies = [
      ...this.state.agent_replies,
      { msg: agentReply, timestamp: Date.now(), isQestion: isQestion },
    ];
    this.setState({
      agent_replies: newAgentReplies,
    });
  }

  render() {
    return (
      <div className="App" style={{ padding: 0 }}>
        <div className="content">
          {this.state.currentView === 1 && (
            <div>
              <div>
                <p>
                  Enter the command to the assistant in the input box below,
                  then press 'Enter'.
                </p>
              </div>
              <div className="center">
                <div className="chat">
                  <div className="time">
                    Assistant is{" "}
                    {this.state.connected === true ? (
                      <span style={{ color: "green" }}>connected</span>
                    ) : (
                      <span style={{ color: "red" }}>not connected</span>
                    )}
                  </div>
                  <div className="messages">
                    <div className="messsages-content" id="scrollbar">
                      <ul className="messagelist" id="chat">
                        {this.renderChatHistory()}
                      </ul>
                      <div
                        style={{ float: "left", clear: "both" }}
                        ref={(el) => {
                          this.messagesEnd = el;
                        }}
                      ></div>
                    </div>
                  </div>
                  <div className="input">
                    <input
                      id="msg"
                      placeholder={
                        this.state.disableInput
                          ? `Waiting for Assistant${this.state.ellipsis}`
                          : "Type your command here"
                      }
                      type="text"
                      disabled={this.state.disableInput}
                    />
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={this.issueStopCommand.bind(this)}
                      className="stop-button"
                      disabled={this.state.disableStopButton}
                    >
                      Stop
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
}

export default InteractApp;
