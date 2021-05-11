/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * Question.js handles the template for the question and answers
 */

import React, { Component } from "react";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";

import "./Question.css";

class Question extends Component {
  constructor(props) {
    super(props);

    this.state = {
      view: 0,
      // asr: false,
      // adtt: false,
      // adtt_text: "",
      parsing_error: false,
      action_dict: {},
      // new_action_dict: {},
      feedback: "",
    };
  }

  finishQuestions() {
    /*
     * finished answering the questions
     * send the data collected to backend to handle storing it
     * go back to the messages page.
     */
    var data = {
      action_dict: this.state.action_dict,
      // new_action_dict: this.state.new_action_dict,
      // asr: this.state.asr,
      // adtt: this.state.adtt,
      parsing_error: this.state.parsing_error,
      // adtt_text: this.state.adtt_text,
      msg: this.props.chats[this.props.failidx].msg,
      feedback: this.state.feedback,
    };

    // Emit socket.io event to save data to error logs
    this.props.stateManager.socket.emit("saveErrorDetailsToCSV", data);

    // go back to message page after writing to database
    this.props.goToMessage();
  }

  saveFeedback(event) {
    //save feedback in state
    this.setState({ feedback: event.target.value });
  }

  renderParsingFail() {
    return (
      <div>
        <h3>
          {" "}
          Thanks for letting me know that I didn't understand the command right.{" "}
        </h3>
        <Button
          variant="contained"
          color="primary"
          onClick={this.finishQuestions.bind(this)}
        >
          Done
        </Button>
      </div>
    );
  }

  renderParsingSuccess() {
    return (
      <div>
        <h3>
          {" "}
          Okay, looks like I understood your command but couldn't follow it all
          the way through. Tell me more about what I did wrong :{" "}
        </h3>
        <TextField
          style={{
            backgroundColor: "white",
          }}
          id="outlined-uncontrolled"
          label=""
          margin="normal"
          variant="outlined"
          onChange={(event) => this.saveFeedback(event)}
        />
        <div></div>
        <Button
          variant="contained"
          color="primary"
          onClick={this.finishQuestions.bind(this)}
        >
          Done
        </Button>
      </div>
    );
  }
  renderEnd() {
    //end screen, user can put any additional feedback
    return (
      <div>
        <h3> Thanks! Submit any other feedback here: </h3>
        <TextField
          style={{
            backgroundColor: "white",
          }}
          id="outlined-uncontrolled"
          label=""
          margin="normal"
          variant="outlined"
          onChange={(event) => this.saveFeedback(event)}
        />
        <div></div>
        <Button
          variant="contained"
          color="primary"
          onClick={this.finishQuestions.bind(this)}
        >
          Done
        </Button>
      </div>
    );
  }

  // renderASRQuestion() {
  //   //check if ASR model output was correct
  //   return (
  //     <div className="question" >
  //       <h5>Is "{this.props.failmsg}" what you said to the bot?</h5>
  //       <List className="answers" component='nav'>
  //         <ListItem button onClick={() => this.answerASR(0)}><ListItemText primary="Yes" /></ListItem>
  //         <ListItem button onClick={() => this.answerASR(1)}><ListItemText primary="No" /></ListItem>
  //       </List>
  //     </div>
  //   );
  // }

  // answerASR(index) {
  //   //handles after the user submits the answer (y/n) to if asr errored or not
  //   if (index === 1) { //if no
  //     this.setState({ asr: true, view: 3 });
  //   } else {
  //     if (this.state.adtt_text === "") {
  //       if (Object.keys(this.state.action_dict).length === 0) {
  //         this.setState({ view: 3 });
  //       } else {
  //         //action dict not empty so show labeling page
  //         this.setState({ view: 2 })
  //       }
  //     } else {
  //       this.setState({ view: 1 })
  //     }
  //   }
  // }

  // renderADTTQuestion() {
  //   //check if ADTT output was correct
  //   return (
  //     <div className="question" >
  //       <h5>Is "{this.state.adtt_text}" what you wanted the bot to do?</h5>
  //       <List className="answers" component='nav'>
  //         <ListItem button onClick={() => this.answerParsing(0)}><ListItemText primary="Yes" /></ListItem>
  //         <ListItem button onClick={() => this.answerParsing(1)}><ListItemText primary="No" /></ListItem>
  //       </List>
  //     </div>
  //   );
  // }

  renderSemanticParserErrorQuestion() {
    /* check if the parser was right.
    Yes -> correct, go to any feedback page
    No -> mark as aprsing error.
    */

    // this.state.action_dict = chatResponses[chatMsg];
    if (this.state.action_dict) {
      if ("dialogue_type" in this.state.action_dict) {
        var dialogue_type = this.state.action_dict.dialogue_type;
        var question_word = "";
        if (dialogue_type === "HUMAN_GIVE_COMMAND") {
          // handle composite action

          // get the action type
          var action_dict = this.state.action_dict.action_sequence[0];
          var action_type = action_dict.action_type.toLowerCase();
          question_word = "to " + action_type + " ";
          // action is build
          if (["build", "dig"].indexOf(action_type) >= 0) {
            if ("schematic" in action_dict) {
              question_word =
                question_word + "'" + action_dict.schematic.text_span + "'";
            }
            if ("location" in action_dict) {
              question_word =
                question_word +
                " at location '" +
                action_dict.location.text_span +
                "'";
            }
            question_word = question_word + " ?";
          } else if (
            ["destroy", "fill", "spawn", "copy", "get", "scout"].indexOf(
              action_type
            ) >= 0
          ) {
            if ("reference_object" in action_dict) {
              question_word =
                question_word +
                "'" +
                action_dict.reference_object.text_span +
                "'";
            }
            if ("location" in action_dict) {
              question_word =
                question_word +
                " at location '" +
                action_dict.location.text_span +
                "'";
            }
            question_word = question_word + " ?";
          } else if (["move"].indexOf(action_type) >= 0) {
            if ("location" in action_dict) {
              question_word =
                question_word +
                " at location '" +
                action_dict.location.text_span +
                "'";
            }
            question_word = question_word + " ?";
          } else if (["stop", "resume", "undo"].indexOf(action_type) >= 0) {
            if ("target_action_type" in action_dict) {
              question_word =
                question_word +
                " at location '" +
                action_dict.target_action_type +
                "'";
            }
            question_word = question_word + " ?";
          }
        } else if (dialogue_type === "GET_MEMORY") {
          // you asked the bot a question
          question_word =
            "to answer a question about something in the Minecraft world ?";
        } else if (dialogue_type === "PUT_MEMORY") {
          // you were trying to teach the bot something
          question_word = "to remember or learn something you taught it ?";
        } else if (dialogue_type === "NOOP") {
          // no operation was requested.
          question_word = "to do nothing ?";
        }
      } else {
        // NOTE: This should never happen ...
        question_word = "did you want me to do nothing ?";
      }
    }
    return (
      <div className="question">
        <h5>Did you want the assistant {question_word}</h5>
        <List className="answers" component="nav">
          <ListItem button onClick={() => this.answerParsing(1)}>
            <ListItemText primary="Yes" />
          </ListItem>
          <ListItem button onClick={() => this.answerParsing(2)}>
            <ListItemText primary="No" />
          </ListItem>
        </List>
      </div>
    );
  }

  answerParsing(index) {
    //handles after the user submits the answer (y/n) to if asr errored or not
    // if (index === 1) { //no of adtt, ask to annotate the tree, set Labeling tool view.
    //   this.setState({ adtt: true, view: 2 });
    // } else
    if (index === 1) {
      // yes, so not a parsing error
      this.setState({ view: 2 });
    } else if (index === 2) {
      // no, so parsing error
      this.setState({ parsing_error: true, view: 1 });
    }
  }

  goToEnd(new_action_dict) {
    //go to the last feedback page and save the new dict from labeling
    this.setState({ view: 3, new_action_dict: new_action_dict });
  }

  componentDidMount() {
    var chatResponses = this.props.stateManager.memory.chatResponse;
    var chatMsg = this.props.chats[this.props.failidx].msg;
    this.setState({ action_dict: chatResponses[chatMsg] });
  }

  render() {
    return (
      <div>
        <div className="msg-header">
          Message you sent to the bot: <br></br>
          {this.props.chats[this.props.failidx].msg}
        </div>
        {/* {this.state.view === 0 ?  this.renderASRQuestion() : null} */}
        {this.state.view === 0
          ? this.renderSemanticParserErrorQuestion()
          : null}
        {this.state.view === 1 ? this.renderParsingFail() : null}
        {this.state.view === 2 ? this.renderParsingSuccess() : null}
        {/* {this.state.view === 1 ? this.renderADTTQuestion() : null} */}
        {/* {this.state.view === 2 ? <Labeling action_dict={this.state.action_dict} message={(this.props.chats[this.props.failidx]).msg} goToEnd={this.goToEnd.bind(this)} /> : null} */}
        {this.state.view === 3 ? this.renderEnd() : null}
      </div>
    );
  }
}

export default Question;
