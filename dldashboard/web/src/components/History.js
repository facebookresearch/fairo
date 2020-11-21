/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from "react";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import "./History.css";

class History extends React.Component {
  renderChatHistory() {
    //render the HTML for the chatHistory with a unique key value
    return this.props.stateManager.memory.chats.map((value, idx) =>
      React.cloneElement(
        <ListItem alignItems="flex-start">
          <div className="chatItem">
            {value.msg !== "" ? <ListItemText primary={value.msg} /> : null}
          </div>
        </ListItem>,
        {
          key: idx.toString(),
        }
      )
    );
  }

  componentDidMount() {
    if (this.props.stateManager) this.props.stateManager.connect(this);
  }

  render() {
    return (
      <div className="history">
        <p>The history of the last 5 commands sent to the bot.</p>
        <List>{this.renderChatHistory()}</List>
      </div>
    );
  }
}

export default History;
