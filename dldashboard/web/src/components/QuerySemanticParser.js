/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React from "react";
import { CopyBlock, dracula } from "react-code-blocks";
import Button from "@material-ui/core/Button";

class QuerySemanticParser extends React.Component {
  constructor(props) {
    super(props);
    this.initialState = {
      action_dict: {},
    };
    this.state = this.initialState;
    this.handleSubmit = this.handleSubmit.bind(this);
    this.querySemanticParser = this.querySemanticParser.bind(this);
    this.qRef = React.createRef();
  }

  querySemanticParser(res) {
    this.setState({
      action_dict: res.action_dict,
    });
  }

  handleSubmit(event) {
    var command = document.getElementById("command_input").innerHTML;
    this.props.stateManager.socket.emit("queryParser", {
      chat: command,
    });
    event.preventDefault();
  }

  componentDidMount() {
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      this.props.stateManager.socket.on(
        "renderActionDict",
        this.querySemanticParser
      );
    }
  }

  render() {
    return (
      <div ref={this.qRef}>
        <p>
          Enter the command below and click on the button to get the parser
          output.
        </p>
        <div style={{ position: "relative", marginLeft: "20%" }}>
          <div
            id="command_input"
            contentEditable="true"
            style={{
              backgroundColor: "white",
              width: "55%",
              height: "30%",
              color: "black",
              marginBottom: "5%",
            }}
          />
          <Button
            style={{ marginLeft: "10%" }}
            size="small"
            variant="contained"
            color="default"
            onClick={this.handleSubmit.bind(this)}
          >
            Get the program
          </Button>
        </div>
        <CopyBlock
          text={JSON.stringify(this.state.action_dict, null, 1)}
          language="python"
          wrapLines
          theme={dracula}
        />
      </div>
    );
  }
}

export default QuerySemanticParser;
