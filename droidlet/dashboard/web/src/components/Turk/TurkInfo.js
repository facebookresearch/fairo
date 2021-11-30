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
      creativityScores: [null],
      diversityScores: [null],
      avgDiversity: null,
      avgCreativity: null,
      quantityScore: null,
      stoplightScore: null,
      commandList: [],
      performanceIndicator: [false, false, false],
      feedback: "",
      commandCorpus: [],
    };
    this.calcCreativity = this.calcCreativity.bind(this);
  }

  handleClick = () => {
    if (this.state.isTimerOn) {
      window.parent.postMessage(JSON.stringify({ msg: "timerOFF" }), "*");
      window.parent.postMessage(JSON.stringify({ msg: {
        interactionScores: {
          creativity: this.state.avgCreativity,
          diversity: this.state.avgDiversity,
          quantity: this.state.quantityScore,
          stoplight: this.state.stoplightScore
        }
      }}), "*");
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
    let creativityScore = 0;
    this.state.commandCorpus.forEach((cmd) => {
      let comparisonCommandArray = removeStopwords(cmd.split(' '));
      let lengthNorm = (commandLength > comparisonCommandArray.length) ? commandLength : comparisonCommandArray.length;
      // Normalize so that the final score is nominally 0-10 where higher is better
      creativityScore += minimumEditDistance.diff(commandArray, comparisonCommandArray).distance / (lengthNorm * corpus_size * 0.1);
    });
    // Most commands will have little overlap, so we need to amplify the signal
    creativityScore = (creativityScore - 9.4) * 16.667
    creativityScore -= Math.pow(Math.abs(optimalCommandLength - commandLength), 2) * 1.5;  // Take off points for long or short commands
    creativityScore = Math.max(1, Math.round(creativityScore*100) / 100);
    console.log("This Creativity Score: " + creativityScore + "/10");
    
    // Save the score and determine avg creativity throughout the HIT
    let newCreativityScores = [...this.state.creativityScores]
    newCreativityScores.push(creativityScore)
    let avgCreativity = newCreativityScores.slice(1,).reduce((a, b) => a + b) / newCreativityScores.slice(1,).length;
    console.log("Avg Creativity Score: " + avgCreativity + "/10");

    // Determine deversity against previously issued commands
    let newCommandList = [...this.state.commandList];
    let diversityScore = 0;
    if (newCommandList.length > 0) {
      newCommandList.forEach((cmd) => {
        let lengthNorm = (commandLength > cmd.length) ? commandLength : cmd.length;
        diversityScore += minimumEditDistance.diff(commandArray, cmd).distance / (lengthNorm * newCommandList.length * 0.1);
      });
    }
    diversityScore = (diversityScore - 6.667) * 3  // Again this is a low bar, need to amplify the signal
    diversityScore = Math.max(1, Math.round(diversityScore*100) / 100);
    console.log("This Diversity Score: " + diversityScore + "/10");

    // Save the score and determine avg diversity throughout the HIT
    let newDiversityScores = [...this.state.diversityScores]
    newDiversityScores.push(diversityScore)
    let avgDiversity = newDiversityScores.slice(1,).reduce((a, b) => a + b) / newDiversityScores.slice(1,).length;
    console.log("Avg Diversity Score: " + avgDiversity + "/10");

    // Save the command and determine score based on number of commands issued
    newCommandList.push(commandArray);
    let quantityScore = Math.min(newCommandList.length, 10);
    console.log("Quantity Score: " + quantityScore + "/10");

    // Scores rise logarithmically, incentivizing meeting a minimum bar in all three areas
    let stoplightScore = (Math.log10(avgCreativity) + Math.log10(avgDiversity) + Math.log10(quantityScore)) / 0.3;
    console.log("Stoplight Score: " + stoplightScore + "/10");
    
    let performance;
    if (stoplightScore < 4.5) performance = [false, false, true];
    else if (stoplightScore < 6.5) performance = [false, true, false];
    else performance = [true, false, false];

    // Generate text feedback for the user
    let feedback = "";
    if (commandLength < (optimalCommandLength - 1)) feedback += "Try sending longer commands. ";
    else if (commandLength > (optimalCommandLength + 1)) feedback += "Try sending shorter commands. ";
    else if (creativityScore < 5) feedback += "Try sending more creative commands. ";
    if (quantityScore < 5) feedback += "Please send more commands. ";
    else if (diversityScore < 4) feedback += "Commands should be more different from one another. ";
    if (feedback.length === 0) feedback += "Good job!";
    
    this.setState({
      creativityScores: newCreativityScores,
      diversityScores: newDiversityScores,
      avgCreativity: avgCreativity,
      avgDiversity: avgDiversity,
      quantityScore: quantityScore,
      stoplightScore: stoplightScore,
      commandList: newCommandList,
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
              <p>Feedback: {this.state.feedback}</p>
            </div>
            {this.state.isSessionEnd ? (
              <p style={{ fontSize: 40, lineHeight: "40px" }}>
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
