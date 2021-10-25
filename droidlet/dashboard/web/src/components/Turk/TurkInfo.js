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
      window.parent.postMessage(JSON.stringify({ msg: "timerOFF" }), "*");
      this.setState({
        isTimerOn: false,
        isSessionEnd: true,
      });
      this.props.stateManager.socket.emit("terminateAgent", {
        turk_experiment_id: this.props.stateManager.getTurkExperimentId(),
        mephisto_agent_id: this.props.stateManager.getMephistoAgentId(),
        turk_worker_id: this.props.stateManager.getTurkWorkerId(),
      });
    } else {
      window.parent.postMessage(JSON.stringify({ msg: "timerON" }), "*");
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
                <div style={{ fontSize: 40 }}>
                  {minutes} : {seconds}
                </div>
                <br />
                <Button
                  className="MsgButton"
                  variant="contained"
                  color={this.state.isTimerOn ? "secondary" : "primary"}
                  onClick={this.handleClick.bind(this)}
                  disabled={this.state.isTimerOn ? minutes < 5 : false}
                >
                  {this.state.isTimerOn ? "End" : "Start"}
                </Button>
                <br />
                {this.state.isTimerOn ? (
                  <div>
                    {minutes < 5 ? (
                      <div>
                        <p>Please interact with the assistant.</p>
                        <p>
                          The 'End' button will appear when 5 minutes have
                          passed.
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p>
                          When you've finished interacting with the assistant,
                          press the 'End' button to proceed to next steps.
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div>
                    <p>Please click on the button to start the session. </p>
                    <p>
                      When at least 5 minutes have passed and you are finished,
                      click on the button to end the session and proceed to next
                      steps.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </ThemeProvider>
    );
  }
}
export default TurkInfo;
