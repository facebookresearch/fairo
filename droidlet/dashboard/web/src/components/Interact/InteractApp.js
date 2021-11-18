/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * App.js handles displaying/switching between different views (settings, message, and TODO: fail)
 */
import React, { Component } from "react";
import "./InteractApp.css";
import Message from "./Message";
import Question from "./Question";
import AgentThinking from "./AgentThinking";

class InteractApp extends Component {
  constructor(props) {
    super(props);
    this.initialState = {
      currentView: 1,
      lastChatActionDict: "",
      status: "",
      chats: [{msg: "", timestamp: Date.now()}],
      failidx: -1,
      agent_replies: [],
      last_reply: "",
    };
    this.state = this.initialState;
    this.goToQuestionWindow = this.goToQuestionWindow.bind(this);
    this.MessageRef = React.createRef();
  }

  setInteractState(chat) {
    // make a shallow copy of chats
    var new_chats = [...this.state.chats];
    new_chats.push(chat);
    this.setState({ chats: new_chats });
    console.log(this.state.chats)
  }

  componentWillUnmount() {
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
  }

  getUrlParameterByName(name) {
    var match = RegExp("[?&]" + name + "=([^&]*)").exec(window.location.search);
    return match && decodeURIComponent(match[1].replace(/\+/g, " "));
  }

  goToMessage() {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(JSON.stringify({ msg: "goToMessaage" }), "*");

    //change the state to switch the view to show Message and save the user input necessary for socket connection
    this.setState({
      currentView: 1,
    });
  }

  goToAgentThinking() {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(JSON.stringify({ msg: "goToAgentThinking" }), "*");

    //change the state to switch the view to show AgentThinking window
    this.setState({
      currentView: 3,
    });
  }

  goToQuestion(idx) {
    // Wait for the logical form of last chat and show the Fail page
    this.props.stateManager.socket.on("setLastChatActionDict", this.goToQuestionWindow);

    // Send request to retrieve the logic form of last sent command
    this.props.stateManager.socket.emit(
      "getChatActionDict",
      this.state.chats[idx]["msg"]
    );
  }

  goToQuestionWindow() {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(JSON.stringify({ msg: "goToQuestion" }), "*");
    // Don't proliferate sio listeners
    this.props.stateManager.socket.off("setLastChatActionDict", this.goToQuestionWindow);
    const replies_len = this.props.stateManager.memory.agent_replies.length;
    const chats_len = this.state.chats.length;
    this.setState({
      agent_replies: this.props.stateManager.memory.agent_replies,
      last_reply: this.props.stateManager.memory.agent_replies[replies_len-1].msg,
      currentView: 2,
      chats: this.state.chats,
      failidx: chats_len-1,
    });
  }

  render() {
    return (
      <div className="App" style={{padding: 0}}>
        <div className="content">
          {this.state.currentView === 1 ? (
            <Message
              status={this.state.status}
              stateManager={this.props.stateManager}
              isMobile={this.props.isMobile}
              ref={this.MessageRef}
              chats={this.state.chats}
              enableVoice={false} // Right now this is hard coded for this branch, should move to stateManager
              agent_replies={this.state.agent_replies}
              goToQuestion={this.goToQuestion.bind(this)}
              goToAgentThinking={this.goToAgentThinking.bind(this)}
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
              agent_reply={this.state.last_reply}
            />
          ) : null}
          {this.state.currentView === 3 ? (
            <AgentThinking
              stateManager={this.props.stateManager}
              chats={this.state.chats}
              goToMessage={this.goToMessage.bind(this)}
              goToQuestion={this.goToQuestion.bind(this)}
              setInteractState={this.setInteractState.bind(this)}
            />
          ) : null}
        </div>
      </div>
    );
  }
}

export default InteractApp;
