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
import KeyboardVoiceIcon from "@material-ui/icons/KeyboardVoice";

import "./Message.css";

const recognition = new window.webkitSpeechRecognition();
recognition.lang = "en-US";

class Message extends Component {
  constructor(props) {
    super(props);
    this.state = {
      recognizing: false,
    };

    this.toggleListen = this.toggleListen.bind(this);
    this.listen = this.listen.bind(this);
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

  handleKeyPress(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      this.handleSubmit();
    }
  }

  componentDidMount() {
    document.addEventListener("keypress", this.handleKeyPress.bind(this));
  }

  toggleListen() {
    //update the variable and call listen
    console.log("togglelisten");
    this.setState({ recognizing: !this.state.recognizing }, this.listen);
  }

  listen() {
    //start listening and grab the output form ASR model to display in textbox
    if (this.state.recognizing) {
      recognition.start();
    } else {
      recognition.stop();
    }
    recognition.onresult = function (event) {
      let msg = "";
      for (var i = 0; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          msg += event.results[i][0].transcript;
        }
      }
      document.getElementById("msg").innerHTML = msg;
    };

    recognition.onerror = (event) => {
      console.log("Error in recognition: " + event.error);
    };
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
        <p>
          Enter the command to the bot in the input box below, or click the mic
          button to start/stop voice input.
        </p>
        <p>
          Click the x next to the message if the outcome wasn't as expected.
        </p>
        <KeyboardVoiceIcon
          className="ASRButton"
          variant="contained"
          color={this.state.recognizing ? "default" : "secondary"}
          fontSize="large"
          onClick={this.toggleListen.bind(this)}
        ></KeyboardVoiceIcon>
        <List>{this.renderChatHistory()}</List>
        <div
          contentEditable="true"
          className="Msg single-line"
          id="msg"
          suppressContentEditableWarning={true}
        >
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

        <p id="callbackMsg">{this.props.status}</p>
        <p id="assistantReply">{this.props.agent_reply} </p>
      </div>
    );
  }
}

export default Message;
