/*
   Copyright (c) Facebook, Inc. and its affiliates.

 * Question.js handles the template for the question and answers in error labeling
 * The error labeling logical tree is as follows:
                      Outcome As Expected?
                             /   \
               No Error <- Yes    No
                                  |
                        Did Agent Understand?
                               /   \
                             Yes    No -> NSP Error
                              |
                    Location Ref Exists?
                            /   \
                          Yes    No -> Other Error
                           |
             Found Location Ref in Memory?
                         /   \
                       Yes    No -> Perception Error
                        |
           Is 'This' the Location Ref?
                     /   \
    Other Error <- Yes    No -> Perception Error
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
      parsing_error: false,
      perception_error: false,
      task_error: false,
      action_dict: {},
      feedback: "",
    };
  }

  componentDidMount() {
    this.props.stateManager.memory.commandState = "idle";
    var lastChatActionDict = this.props.stateManager.memory.lastChatActionDict;
    this.setState({
      action_dict: lastChatActionDict,
      agent_reply: this.props.stateManager.memory.last_reply,
    });
  }

  renderActionQuestion() {
    return (
      <div>
        <h3> Did I successfully do the task you asked me to complete? </h3>
        <List className="answers" component="nav">
          <ListItem button onClick={() => this.answerAction(1)}>
            <ListItemText className="listButton" primary="Yes" />
          </ListItem>
          <ListItem button onClick={() => this.answerAction(2)}>
            <ListItemText className="listButton" primary="No" />
          </ListItem>
        </List>
      </div>
    );
  }

  answerAction(index) {
    //answer to question #1
    //handles after the user submits the answer (y/n) to if the agent task was correct
    if (index === 1) {
      // yes, no error, go to end
      this.setState({ view: 5 });
    } else if (index === 2) {
      // no, error exists, go to parser question
      this.setState({ view: 1 });
    }
  }

  renderParserQuestion() {
    /* check if the parser was right.*/
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
          // action is build or dig
          if (["build", "dig"].indexOf(action_type) >= 0) {
            if ("schematic" in action_dict) {
              if ("text_span" in action_dict.schematic) {
                question_word =
                  question_word + "'" + action_dict.schematic.text_span + "'";
              }
              // If we don't have a convenient text_span, find the words referenced by index
              else if ("where_clause" in action_dict.schematic.filters) {
                let qty = "";
                if ("selector" in action_dict.schematic.filters) {
                  qty = action_dict.schematic.filters.selector.ordinal;
                }
                let antecedent = [qty, "", "", "", ""]; // qty then size then colour then block type then name. Ignore everything else.
                action_dict.schematic.filters.where_clause.AND.forEach(
                  (clause) => {
                    if (clause.pred_text === "has_size")
                      antecedent[1] = clause.obj_text;
                    else if (clause.pred_text === "has_colour")
                      antecedent[2] = clause.obj_text;
                    else if (clause.pred_text === "has_block_type")
                      antecedent[3] = clause.obj_text;
                    else if (clause.pred_text === "has_name")
                      antecedent[4] = clause.obj_text;
                  }
                );
                question_word =
                  question_word +
                  "'" +
                  antecedent.join(" ").replace(/  +/g, " ").trim() +
                  "'";
              }
            }
            if ("location" in action_dict) {
              if ("text_span" in action_dict.location) {
                question_word =
                  question_word +
                  " at location '" +
                  action_dict.location.text_span +
                  "'";
              } else {
                question_word = question_word + " at this location "; // Not worth it to handle all of the potential references?
              }
            }
            question_word = question_word + " ?";
          } else if (
            [
              "destroy",
              "fill",
              "spawn",
              "copy",
              "get",
              "scout",
              "freebuild",
            ].indexOf(action_type) >= 0
          ) {
            if ("reference_object" in action_dict) {
              if ("text_span" in action_dict.reference_object) {
                question_word =
                  question_word +
                  "'" +
                  action_dict.reference_object.text_span +
                  "'";
              }
              // If we don't have a convenient text_span, find the words referenced by index
              else if ("where_clause" in action_dict.reference_object.filters) {
                let qty = "";
                if ("selector" in action_dict.reference_object.filters) {
                  qty = action_dict.reference_object.filters.selector.ordinal;
                }
                let antecedent = [qty, "", "", "", ""]; // qty then size then colour then block type then name. Ignore everything else.
                action_dict.reference_object.filters.where_clause.AND.forEach(
                  (clause) => {
                    if (clause.pred_text === "has_size")
                      antecedent[1] = clause.obj_text;
                    else if (clause.pred_text === "has_colour")
                      antecedent[2] = clause.obj_text;
                    else if (clause.pred_text === "has_block_type")
                      antecedent[3] = clause.obj_text;
                    else if (clause.pred_text === "has_name")
                      antecedent[4] = clause.obj_text;
                  }
                );
                question_word =
                  question_word +
                  "'" +
                  antecedent.join(" ").replace(/  +/g, " ").trim() +
                  "'";
              }
            }
            if ("location" in action_dict) {
              if ("text_span" in action_dict.location) {
                question_word =
                  question_word +
                  " at location '" +
                  action_dict.location.text_span +
                  "'";
              } else {
                question_word = question_word + " at this location ";
              }
            }
            question_word = question_word + " ?";
          } else if (["move"].indexOf(action_type) >= 0) {
            if ("location" in action_dict) {
              if ("text_span" in action_dict.location) {
                question_word =
                  question_word +
                  " to location '" +
                  action_dict.location.text_span +
                  "'";
              } else {
                question_word = question_word + " to here";
              }
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
          } else if (["otheraction"].indexOf(action_type) >= 0) {
            question_word =
              "to perform an action not in assistant capabilities ?";
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
          question_word = "to just respond verbally with no action ?";
        }
      } else {
        // NOTE: This should never happen ...
        question_word = "to do nothing ?";
      }
    } else {
      // shouldn't happen
      return (
        <div>
          <h3> Thanks! Press to continue.</h3>
          <Button
            variant="contained"
            color="primary"
            onClick={() => this.props.goToMessage()}
          >
            Done
          </Button>
        </div>
      );
    }
    return (
      <div className="question">
        <h3>Did you want the assistant {question_word}</h3>
        <List className="answers" component="nav">
          <ListItem button onClick={() => this.answerParsing(1)}>
            <ListItemText className="listButton" primary="Yes" />
          </ListItem>
          <ListItem button onClick={() => this.answerParsing(2)}>
            <ListItemText className="listButton" primary="No" />
          </ListItem>
          <ListItem button onClick={() => this.setState({ view: 0 })}>
            <ListItemText className="listButton" primary="Go Back" />
          </ListItem>
        </List>
      </div>
    );
  }

  answerParsing(index) {
    //handles after the user submits the answer (y/n) to if NSP errored or not
    if (index === 1) {
      // yes, so not a parsing error
      this.evalCommandPerception();
      this.setState({ view: 3 });
    } else if (index === 2) {
      // no, so parsing error
      this.setState({ parsing_error: true, view: 2 });
    }
  }

  renderParsingFail() {
    return (
      <div>
        <h3>
          {" "}
          Thanks for letting me know that I didn't understand the command right.{" "}
        </h3>
        <List className="answers" component="nav">
          <ListItem button onClick={this.finishQuestions.bind(this)}>
            <ListItemText className="listButton" primary="Done" />
          </ListItem>
          <ListItem button onClick={() => this.setState({ view: 1 })}>
            <ListItemText className="listButton" primary="Go Back" />
          </ListItem>
        </List>
      </div>
    );
  }

  renderVisionFail() {
    return (
      <div>
        <h3>
          {" "}
          Thanks for letting me know that I didn't detect the object right.{" "}
        </h3>
        <List className="answers" component="nav">
          <ListItem button onClick={this.finishQuestions.bind(this)}>
            <ListItemText className="listButton" primary="Done" />
          </ListItem>
          <ListItem button onClick={() => this.setState({ view: 3 })}>
            <ListItemText className="listButton" primary="Go Back" />
          </ListItem>
        </List>
      </div>
    );
  }

  check_reference_object_in_action_dict(action) {
    var action_dict = action;
    for (var key in action_dict) {
      if (key == "reference_object") {
        return true;
      } else {
        if (action_dict[key].constructor == Object) {
          if (this.check_reference_object_in_action_dict(action_dict[key])) {
            return true;
          }
        }
      }
    }
    return false;
  }

  extractLocationRef(action_dict) {
    var locationRef = "";
    if ("reference_object" in action_dict) {
      if ("text_span" in action_dict.reference_object) {
        locationRef = "'" + action_dict.reference_object.text_span + "'";
      }
      // If we don't have a convenient text_span, find the words referenced by index
      else if (
        "filters" in action_dict.reference_object &&
        "where_clause" in action_dict.reference_object.filters
      ) {
        let qty = "";
        if ("selector" in action_dict.reference_object.filters) {
          qty = action_dict.reference_object.filters.selector.ordinal;
        }
        let antecedent = [qty, "", "", "", ""]; // qty then size then colour then block type then name. Ignore everything else.
        action_dict.reference_object.filters.where_clause.AND.forEach(
          (clause) => {
            if (clause.pred_text === "has_size")
              antecedent[1] = clause.obj_text;
            else if (clause.pred_text === "has_colour")
              antecedent[2] = clause.obj_text;
            else if (clause.pred_text === "has_block_type")
              antecedent[3] = clause.obj_text;
            else if (clause.pred_text === "has_name")
              antecedent[4] = clause.obj_text;
          }
        );
        locationRef =
          "'" + antecedent.join(" ").replace(/  +/g, " ").trim() + "'";
      }
    }
    return locationRef;
  }

  evalCommandPerception() {
    //                 Reference Object Exists in dictionary ?
    //                         /   \
    //                       Yes    No -> Other Error
    //                        |
    //             Found Location in Memory?
    //                      /   \
    //                    Yes    No -> Perception Error
    let ref_object = false;
    let reference_object_description = null;
    // Check if reference object exists in the dictionary anywhere
    if (this.state.action_dict) {
      if (this.state.action_dict["dialogue_type"] == "HUMAN_GIVE_COMMAND") {
        // also implement for get and put memory
        for (const action of this.state.action_dict.action_sequence) {
          ref_object = this.check_reference_object_in_action_dict(action);
        }
      }

      // If yes, find reference object description.
      if (ref_object == true) {
        const action_dict = this.state.action_dict.action_sequence[0];
        // Check for location at top level and extract the reference text
        let considered_action_dict = null;
        if ("location" in action_dict) {
          considered_action_dict = action_dict.location;
        } else if ("reference_object" in action_dict) {
          considered_action_dict = action_dict;
        }

        if (considered_action_dict) {
          reference_object_description = this.extractLocationRef(
            considered_action_dict
          );
        }
      }
      // If no reference object description found no perception error.
    } else {
      console.log("no action dictionary found ..."); // Shouldn't happen....
    }
    this.setState({
      reference_object_description: reference_object_description,
    });

    // else {
    //   // shouldn't happen
    //   return (
    //     <div>
    //       <h3> Thanks! Press to continue.</h3>
    //       <Button
    //         variant="contained"
    //         color="primary"
    //         onClick={() => this.props.goToMessage()}
    //       >
    //         Done
    //       </Button>
    //     </div>
    //   );
    // }
  }

  renderVisionQuestion() {
    //        Is 'This' the Location Ref?
    //                  /   \
    // Other Error <- Yes    No -> Perception Error
    let reference_object_description = this.state.reference_object_description;
    if (!reference_object_description) {
      // Not perception error.
      this.setState({ view: 6 });
      return;
    }
    return (
      <div className="question">
        <h3>
          Okay, looks like I understood your command. I was looking for an
          object of interest called : {reference_object_description}. Here's
          what I think it is. Does that look right ?
        </h3>
        <List className="answers" component="nav">
          <ListItem button onClick={() => this.answerVision(1)}>
            <ListItemText className="listButton" primary="Yes" />
          </ListItem>
          <ListItem button onClick={() => this.answerVision(2)}>
            <ListItemText className="listButton" primary="No" />
          </ListItem>
          <ListItem button onClick={() => this.setState({ view: 1 })}>
            <ListItemText className="listButton" primary="Go Back" />
          </ListItem>
        </List>
      </div>
    );
  }

  answerVision(index) {
    //handles after the user submits the answer (y/n) to if NSP errored or not
    if (index === 1) {
      // yes, so not a vision error
      this.setState({ view: 6 });
    } else if (index === 2) {
      // no, so vision error
      this.setState({ vision_error: true, view: 4 });
    }
  }

  renderOtherError() {
    return (
      <div>
        <h3>
          {" "}
          Okay, looks like I understood your command but didn't complete it.
          Please tell me more about what I did wrong:{" "}
        </h3>
        <TextField
          id="outlined-uncontrolled"
          label=""
          margin="normal"
          variant="outlined"
          onChange={(event) => this.saveFeedback(event)}
        />
        <List className="answers" component="nav">
          <ListItem button onClick={this.finishQuestions.bind(this)}>
            <ListItemText className="listButton" primary="Done" />
          </ListItem>
          <ListItem button onClick={() => this.setState({ view: 3 })}>
            <ListItemText className="listButton" primary="Go Back" />
          </ListItem>
        </List>
      </div>
    );
  }

  saveFeedback(event) {
    //save feedback in state
    this.setState({ feedback: event.target.value });
  }

  goToEnd(new_action_dict) {
    //go to the last feedback page and save the new dict from labeling
    this.setState({ view: 4, new_action_dict: new_action_dict });
  }

  renderEnd() {
    //end screen, user can put any additional feedback
    return (
      <div>
        <h3> Thanks! Submit any other feedback here (optional): </h3>
        <TextField
          id="outlined-uncontrolled"
          label=""
          margin="normal"
          variant="outlined"
          onChange={(event) => this.saveFeedback(event)}
        />
        <List className="answers" component="nav">
          <ListItem button onClick={this.finishQuestions.bind(this)}>
            <ListItemText className="listButton" primary="Done" />
          </ListItem>
          <ListItem button onClick={() => this.setState({ view: 0 })}>
            <ListItemText className="listButton" primary="Go Back" />
          </ListItem>
        </List>
      </div>
    );
  }

  finishQuestions() {
    /*
     * finished answering the questions
     * send the data collected to backend to handle storing it
     * go back to the messages page.
     */
    var data = {
      action_dict: this.state.action_dict,
      parsing_error: this.state.parsing_error,
      task_error: this.state.task_error,
      msg: this.props.chats[this.props.failidx].msg,
      feedback: this.state.feedback,
    };

    // Emit socket.io event to save data to error logs and Mephisto
    this.props.stateManager.socket.emit("saveErrorDetailsToCSV", data);
    // Parsed: data.msg.data = data
    window.parent.postMessage(JSON.stringify({ msg: data }), "*");
    // go back to message page after writing to database
    this.props.goToMessage();
  }

  render() {
    return (
      <div>
        <div className="msg-header">
          Message you sent to the assistant: <br></br>
          <strong>{this.props.chats[this.props.failidx].msg}</strong>
        </div>
        <div className="msg-header">
          The assistant responded:<br></br>
          <strong>{this.state.agent_reply}</strong>
        </div>
        {this.state.view === 0 ? this.renderActionQuestion() : null}
        {this.state.view === 1 ? this.renderParserQuestion() : null}
        {this.state.view === 2 ? this.renderParsingFail() : null}
        {this.state.view === 3 ? this.renderVisionQuestion() : null}
        {this.state.view === 4 ? this.renderVisionFail() : null}
        {this.state.view === 5 ? this.renderEnd() : null}
        {this.state.view === 6 ? this.renderOtherError() : null}
      </div>
    );
  }
}

export default Question;
