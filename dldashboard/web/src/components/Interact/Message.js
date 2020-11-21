/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * Message.js implements ASR, send the chat message, switch to the fail or back to settings view
 */

import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemSecondaryAction from "@material-ui/core/ListItemSecondaryAction";
import ListItemText from "@material-ui/core/ListItemText";
import FailIcon from "@material-ui/icons/Cancel";

import IconButton from "@material-ui/core/IconButton";

import "./Message.css";

const recognition = new window.webkitSpeechRecognition();
recognition.lang = "en-US";

class Message extends Component {
  constructor(props) {
    super(props);
    this.elementRef = React.createRef();
  }

  renderChatHistory() {
    //render the HTML for the chatHistory with a unique key value
    return this.props.chats.map((value, idx) =>
      React.cloneElement(
        <ListItem>
          <ListItemText primary={value.msg} />
          <ListItemSecondaryAction>
            {value.msg !== "" ? (
              <IconButton
                edge="end"
                aria-label="Fail"
                onClick={() => this.props.goToQuestion(idx)}
              >
                <FailIcon className="cross" />
              </IconButton>
            ) : null}
          </ListItemSecondaryAction>
        </ListItem>,
        {
          key: idx.toString(),
        }
      )
    );
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  handleSubmit() {
    //get the message
    var chatmsg = document.getElementById("msg").innerHTML;
    if (chatmsg.replace(/\s/g, "") !== "") {
      //add to chat history box of parent
      this.props.setInteractState({ msg: chatmsg, failed: false });
      //socket connection
      this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
      //clear the textbox
      document.getElementById("msg").innerHTML = "";
    }
  }

  render() {
    return (
      <div className="Chat">
        {/* <p>Press spacebar to start/stop recording.</p> */}
        <p>Enter the command to the bot in the input box below.</p>
        <p>
          Click the x next to the message if the outcome wasn't as expected.
        </p>
        <List>{this.renderChatHistory()}</List>
        <div contentEditable="true" className="Msg single-line" id="msg">
          {" "}
        </div>
        <Button
          className="MsgButton"
          variant="contained"
          color="primary"
          onClick={this.handleSubmit.bind(this)}
        >
          {" "}
          Submit{" "}
        </Button>
        {}

        <p id="callbackMsg">{this.props.status}</p>
      </div>
    );
  }
}

export default Message;
