/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import muiTheme from "./theme";
import { MuiThemeProvider as ThemeProvider } from "@material-ui/core/styles";
import minimumEditDistance from "minimum-edit-distance";
import {removeStopwords} from "stopword";
import "status-indicator/styles.css";
import "./TurkInfo.css";

class TurkInfo extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTimerOn: false,
      isSessionEnd: false,
      startTime: 0,
      timeElapsed: 0,
      commandScores: [null],
      performanceIndicator: [false, false, false],
      feedback: "",
      commandCorpus: [],
    };
    this.calcCreativity = this.calcCreativity.bind(this);
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

  componentDidMount() {
    this.props.stateManager.connect(this);
    // Download data from S3 and parse into commands
    fetch("https://craftassist.s3.us-west-2.amazonaws.com/pubr/nsp_data.txt")
      .then(response => response.blob())
      .then(blob => blob.text())
      .then(text => text.split('\n'))
      .then((result) => {
        let corpusArray = [];
        result.forEach((res) => {
          corpusArray.push(res.split('|')[0]);
        });
        this.setState({
          commandCorpus: corpusArray,
        });
      });
  }

  componentWillUnmount() {
    this.props.stateManager.disconnect(this);
  }

  calcCreativity(command) {
    const optimalCommandLength = 4;
    const commandArray = removeStopwords(command.split(' '));  // Don't want to reward or penalize lots of stop words
    const commandLength = commandArray.length;
    const corpus_size = this.state.commandCorpus.length;
    let score = 0.0;
    this.state.commandCorpus.forEach((cmd) => {
      let comparisonCommandArray = removeStopwords(cmd.split(' '));
      let lengthNorm = (commandLength > comparisonCommandArray.length) ? commandLength : comparisonCommandArray.length;
      // Normalize so that the final score is nominally 0-10 where higher is better
      score += minimumEditDistance.diff(commandArray, comparisonCommandArray).distance / (lengthNorm * corpus_size * 0.1);
    });
    console.log("Uncorrected Score: " + score + "/10")
    // Most commands will have little overlap, so we need to amplify the signal
    score = (score - 9.5) * 20
    score -= Math.pow(Math.abs(optimalCommandLength - commandLength), 2) * 1.5;  // Take off points for long or short commands
    score = Math.max(0, Math.round(score*100) / 100);

    // Generate text feedback for the user
    let feedback = "";
    if (commandLength < (optimalCommandLength - 1)) feedback += "Try sending longer commands";
    else if (commandLength > (optimalCommandLength + 1)) feedback += "Try sending shorter commands";
    else if (score < 5) feedback += "Try sending more creative commands";
    else feedback += "Good job!";
    
    // Save the score and determine overall performance throughout the HIT
    let newScores = [...this.state.commandScores]
    newScores.push(score)
    let avgScore = newScores.slice(1,).reduce((a, b) => a + b) / newScores.slice(1,).length;
    let performance;
    if (avgScore < 3) performance = [false, false, true];
    else if (avgScore < 5) performance = [false, true, false];
    else performance = [true, false, false];

    this.setState({
      commandScores: newScores,
      feedback: feedback,
      performanceIndicator: performance,
    });
  }

  render() {
    const { timeElapsed } = this.state;
    let seconds = ("0" + (Math.floor(timeElapsed / 1000) % 60)).slice(-2);
    let minutes = ("0" + (Math.floor(timeElapsed / 60000) % 60)).slice(-2);
    return (
      <ThemeProvider theme={muiTheme}>
        <div className="App">
          <div className="infoContent">
            <div className="stoplight">
              <h4>HIT Performance Indicator:</h4>
              {this.state.performanceIndicator[0] ? <status-indicator positive pulse></status-indicator> : <status-indicator positive ></status-indicator>}
              {this.state.performanceIndicator[1] ? <status-indicator intermediary pulse></status-indicator> : <status-indicator intermediary ></status-indicator>}
              {this.state.performanceIndicator[2] ? <status-indicator negative pulse></status-indicator> : <status-indicator negative ></status-indicator>}
              <p>Last Command Score: {this.state.commandScores[this.state.commandScores.length - 1]}/10</p>
              <p>Feedback: {this.state.feedback}</p>
            </div>
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
