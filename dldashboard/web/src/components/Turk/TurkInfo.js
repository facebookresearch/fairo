/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import muiTheme from "./theme";
import { MuiThemeProvider as ThemeProvider } from "@material-ui/core/styles";

class TurkInfo extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTimerOn: false,
      isSessionEnd: false,
      startTime: 0,
      timeElapsed: 0,
    };
  }

  handleClick = () => {
    if (this.state.isTimerOn) {
      this.setState({
        isTimerOn: false,
        isSessionEnd: true,
      });
      this.props.stateManager.socket.emit("terminateAgent", "");
    } else {
      this.setState({
        isTimerOn: true,
        startTime: Date.now(),
      });
      this.timer = setInterval(() => {
        this.setState({
          timeElapsed: Date.now() - this.state.startTime,
        });
      }, 10);
    }
  };

  render() {
    const { timeElapsed } = this.state;
    let seconds = ("0" + (Math.floor(timeElapsed / 1000) % 60)).slice(-2);
    let minutes = ("0" + (Math.floor(timeElapsed / 60000) % 60)).slice(-2);
    return (
      <ThemeProvider theme={muiTheme}>
        <div className="App">
          <div className="content">
            {this.state.isSessionEnd ? (
              <p style={{ fontSize: 40 }}>
                Thanks for interacting with the bot. You may leave the page now.
              </p>
            ) : (
              <div>
                <div className="App-title" style={{ fontSize: 35 }}>
                  Interact with the agent
                </div>
                <p>
                  In this task, you will interact with a bot that is there to
                  help you, You can use the chatbox on the top-left pane to talk
                  to the bot. You can also control you own avatar by clicking on
                  the bottom-left pane and entering the voxel world
                </p>
                <div style={{ fontSize: 30 }}>How to control your avatar?</div>
                <p>
                  Press 'w/a/s/d' to move forward/left/backward/right, 'space'
                  to jump and 'esc' to leave the voxel world and back to the
                  website
                </p>

                <br />
                <br />
                <div style={{ fontSize: 40 }}>
                  {minutes} : {seconds}
                </div>
                <br />
                <Button
                  className="MsgButton"
                  variant="contained"
                  color={this.state.isTimerOn ? "secondary" : "primary"}
                  onClick={this.handleClick.bind(this)}
                >
                  {this.state.isTimerOn ? "End" : "Start"}
                </Button>
                <br />
                <p>Please click on the button to start the session. </p>
                <p>
                  When you finished, click on the button to end the session and
                  proceed to next steps.
                </p>
              </div>
            )}
          </div>
        </div>
      </ThemeProvider>
    );
  }
}
export default TurkInfo;
