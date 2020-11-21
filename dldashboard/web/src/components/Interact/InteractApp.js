/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * App.js handles displaying/switching between different views (settings, message, and TODO: fail)
 */
import React, { Component } from "react";
import "./InteractApp.css";
import Message from "./Message";
import Question from "./Question";

class InteractApp extends Component {
  constructor(props) {
    super(props);
    this.initialState = {
      currentView: 1,
      chatResponse: "",
      status: "",
      chats: [{ msg: "", failed: false }],
      failidx: -1,
    };
    this.state = this.initialState;
    this.MessageRef = React.createRef();
  }

  setInteractState(chat) {
    // make a shallow copy of chats
    var new_chats = [...this.state.chats];
    new_chats.shift();
    new_chats.push(chat);
    this.setState({ chats: new_chats });
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  getUrlParameterByName(name) {
    var match = RegExp("[?&]" + name + "=([^&]*)").exec(window.location.search);
    return match && decodeURIComponent(match[1].replace(/\+/g, " "));
  }

  goToMessage() {
    //change the state to switch the view to show Message and save the user input necessary for socket connection
    var newchats = this.state.chats;
    if (this.state.failidx !== -1) {
      newchats[this.state.failidx].failed = true;
    }
    this.setState({
      currentView: 1,
      chats: newchats,
    });
  }

  goToQuestion(idx) {
    //change the state to switch view to show Fail page
    this.setState({
      currentView: 2,
      chats: this.state.chats,
      failidx: idx,
    });
  }

  render() {
    return (
      <div className="App">
        <div className="content">
          {this.state.currentView === 1 ? (
            <Message
              status={this.state.status}
              stateManager={this.props.stateManager}
              ref={this.MessageRef}
              chats={this.state.chats}
              goToQuestion={this.goToQuestion.bind(this)}
              setInteractState={this.setInteractState.bind(this)}
            />
          ) : null}
          {this.state.currentView === 2 ? (
            <Question
              stateManager={this.props.stateManager}
              chats={this.state.chats}
              failidx={this.state.failidx}
              goToMessage={this.goToMessage.bind(this)}
              failmsg={this.state.chats[this.state.failidx].msg}
            />
          ) : null}
        </div>
      </div>
    );
  }
}

export default InteractApp;
