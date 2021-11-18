/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * Message.js implements ASR, send the chat message, switch to the fail or back to settings view
 */

import React, { Component } from "react";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";

import "./Message.css";

class Message extends Component {
  constructor(props) {
    super(props);
    this.initialState = {
      recognizing: false,
      enableVoice: this.props.enableVoice,
      connected: false,
    };
    this.state = this.initialState;
    this.elementRef = React.createRef();
    this.bindKeyPress = this.handleKeyPress.bind(this); // this is used in keypressed event handling
  }

  renderChatHistory() {
    // Pull in user chats and agent replies from props, filter out any empty ones
    let chats = this.props.chats.filter(chat => chat.msg !== "");
    let replies = this.props.agent_replies.filter(reply => reply.msg !== "");
    chats = chats.filter(chat => chat.msg)
    replies = replies.filter(reply => reply.msg)
    // Label each chat based on where it came from
    chats.forEach(chat => chat['sender'] = 'message user');
    replies.forEach(reply => reply['sender'] = 'message agent');
    // Strip out the 'Agent: ' prefix if it's there
    replies.forEach(function(reply) {
      if (reply['msg'].includes("Agent: ")) {
        reply['msg'] = reply['msg'].substring(7);
      }
    });
    // Zip it into one list, sort by timestamp, and send it off to be rendered
    let chat_history = chats.concat(replies);
    chat_history.sort(function (a, b) { return a.timestamp - b.timestamp; });

    return chat_history.map((chat) =>
      React.cloneElement(
        <li className={chat.sender}>
          {chat.msg}
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
    this.props.stateManager.connect(this);
    document.addEventListener("keypress", this.bindKeyPress);
    this.setState({ connected: this.props.stateManager.connected });
  }

  componentWillUnmount() {
    this.props.stateManager.disconnect(this);
    document.removeEventListener("keypress", this.bindKeyPress);
  }

  handleSubmit() {
    //get the message
    var chatmsg = document.getElementById("msg").value;
    if (chatmsg.replace(/\s/g, "") !== "") {
      //add to chat history box of parent
      this.props.setInteractState({ msg: chatmsg, timestamp: Date.now() });
      //log message to flask
      this.props.stateManager.logInteractiondata("text command", chatmsg);
      //socket connection
      this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
      //update StateManager command state
      this.props.stateManager.memory.commandState = "sent";
      //clear the textbox
      document.getElementById("msg").value = "";
      //clear the agent reply that will be shown in the question pane
      this.props.stateManager.memory.agent_reply = "";
      //change to the AgentThinking view pane
      this.props.goToAgentThinking();
    }
  }

  componentDidUpdate() {
    return
  }

  render() {
    return (
      <div>
        <div>
            <p>Enter the command to the assistant in the input box below, then press 'Enter'.</p>
          </div>
        <div className="center">
          <div className="chat">
            <div className="time">
              Assistant is {this.state.connected === true ? (
                <span style={{color: 'green'}}>connected</span>
              ) : (
                <span style={{color: 'red'}}>not connected</span>
              )}
            </div>
            <div className="messages">
              <ul className="messagelist" id="chat">
                {this.renderChatHistory()}
              </ul>
            </div>
            <div className="input">
              <input id="msg" placeholder="Type your command here" type="text" />
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Message;
