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
      chats: [{ msg: "", failed: false }],
      failidx: -1,
      agent_reply: "",
    };
    this.state = this.initialState;
    this.setAssistantReply = this.setAssistantReply.bind(this);
    this.MessageRef = React.createRef();
  }

  setInteractState(chat) {
    // make a shallow copy of chats
    var new_chats = [...this.state.chats];
    new_chats.shift();
    new_chats.push(chat);
    this.setState({ chats: new_chats });
  }

  setAssistantReply(res) {
    // show assistant's reply in the Message component
    this.setState({
      agent_reply: res.agent_reply,
    });
  }

  componentDidMount() {
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      this.props.stateManager.socket.on(
        "showAssistantReply",
        this.setAssistantReply
      );
    }
  }

  componentWillUnmount() {
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
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

  goToAgentThinking() {
    //change the state to switch the view to show AgentThinking window
    this.setState({
      currentView: 3,
    });
  }

  goToQuestion(idx) {
    // first send request to retrieve the logic form of last sent command before showing NSP Error annotation page to users
    this.props.stateManager.socket.emit(
      "getChatActionDict",
      this.state.chats[idx]["msg"]
    );

    // then wait 3 seconds for the logical form of last chat and show the Fail page (by setting currentView)
    setTimeout(() => {
      this.setState({
        currentView: 2,
        chats: this.state.chats,
        failidx: idx,
      });
    }, 3000);
  }

  render() {
    return (
      <div className="App">
        <div className="content">
          {this.state.currentView === 1 ? (
            <Message
              status={this.state.status}
              stateManager={this.props.stateManager}
              isMobile={this.props.isMobile}
              ref={this.MessageRef}
              chats={this.state.chats}
              enableVoice={false} // Right now this is hard coded for this branch, should move to stateManager
              agent_reply={this.state.agent_reply}
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
              agent_reply={this.state.agent_reply}
            />
          ) : null}
          {this.state.currentView === 3 ? (
            <AgentThinking stateManager={this.props.stateManager} />
          ) : null}
        </div>
      </div>
    );
  }
}

export default InteractApp;
