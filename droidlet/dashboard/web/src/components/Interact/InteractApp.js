import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import "./InteractApp.css";

const ANSWER_ACTION = "answerAction";
const ANSWER_PARSING = "answerParsing";
const ANSWER_VISION = "answerVision";
const CLARIFICATION = "clarification";
const PLEASE_RESUME = " You can type a new command now.";

class InteractApp extends Component {
  allowedStates = [
    "sent",
    "received",
    "thinking",
    "done_thinking",
    "executing",
  ];

  constructor(props) {
    super(props);
    this.initialState = {
      connected: false,
      ellipsis: "",
      commandState: "idle",
      now: null,
      disableInput: false,
      lastChatActionDict: "",
      memory_entries: null,
      chats: [{ msg: "", timestamp: Date.now() }],
      response_options: [],
      last_command: "",
      agent_replies: [{}],
      agentType: null,
      isTurk: false,
      action_dict: {},
      parsing_error: false,
      vision_error: false,
      task_error: false,
      feedback: "",
      isSaveFeedback: false,
      clarify: false,
    };

    this.state = this.initialState;
    this.intervalId = null;
    this.messagesEnd = null;
    this.elementRef = React.createRef();

    this.bindKeyPress = this.handleKeyPress.bind(this);
    this.sendTaskStackPoll = this.sendTaskStackPoll.bind(this);
    this.issueResetCommand = this.issueResetCommand.bind(this);
    this.answerAction = this.answerAction.bind(this);
    this.answerParsing = this.answerParsing.bind(this);
    this.answerVision = this.answerVision.bind(this);
    this.receiveTaskStackPoll = this.receiveTaskStackPoll.bind(this);
  }

  /**********************************************************************************
   ********************************** Component Utils ********************************
   **********************************************************************************/

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  saveFeedback() {
    var data = {
      msg: this.state.last_command,
      action_dict: this.state.action_dict,
      parsing_error: this.state.parsing_error,
      task_error: this.state.task_error,
      vision_error: this.state.vision_error,
      feedback: this.state.feedback,
    };
    // Emit socket.io event to save data to error logs and Mephisto
    this.props.stateManager.socket.emit("saveErrorDetailsToCSV", data);

    this.setState({
      parsing_error: false,
      task_error: false,
      vision_error: false,
      feedback: "",
      disableInput: false,
    });
  }

  removeButtonsFromLastQuestion() {
    var new_agent_replies = [...this.state.agent_replies];
    new_agent_replies.forEach(
      (agent_reply) => (
        (agent_reply.isQuestion = false), (agent_reply.enableBack = false)
      )
    );
    this.setState({ agent_replies: new_agent_replies });
  }

  componentDidMount() {
    document.addEventListener("keypress", this.bindKeyPress);
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      var lastChatActionDict =
        this.props.stateManager.memory.lastChatActionDict;
      var memory_entries = this.props.stateManager.memory.memory_entries;
      this.setState({
        isTurk: this.props.stateManager.memory.isTurk,
        agent_replies: this.props.stateManager.memory.agent_replies,
        connected: this.props.stateManager.connected,
        action_dict: lastChatActionDict,
        memory_entries: memory_entries,
        // mockup data for other question case
        // action_dict: {}
      });
    }
    // Scroll messsage panel to bottom
    this.scrollToBottom();
  }

  componentWillUnmount() {
    document.removeEventListener("keypress", this.bindKeyPress);
    if (this.props.stateManager) this.props.stateManager.disconnect(this);
  }

  /************************************************************************************
   *********************************** Messaging ***************************************
   ************************************************************************************/

  addNewAgentReplies({
    msg,
    isQuestion,
    questionType,
    disablePreviousAnswer,
    enableBack,
  }) {
    // Clear any lingering status messages before saving
    this.setState({
      agent_replies: this.props.stateManager.memory.agent_replies,
    });

    const { agent_replies } = this.state;
    let new_agent_replies = disablePreviousAnswer
      ? agent_replies.map((item) => ({
          ...item,
          isQuestion: false,
          enableBack: false,
        }))
      : agent_replies;
    new_agent_replies = [
      ...new_agent_replies,
      {
        msg: msg,
        timestamp: Date.now() + 1,
        questionType: questionType,
        isQuestion: isQuestion,
        enableBack: enableBack,
      },
    ];
    this.setState({
      agent_replies: new_agent_replies,
    });
    this.props.stateManager.memory.agent_replies = new_agent_replies;
  }

  updateChat(chat) {
    // make a shallow copy of chats
    var new_chats = [...this.state.chats];
    new_chats.push(chat);
    this.setState({ chats: new_chats });
  }

  renderChatHistory() {
    // Pull in user chats and agent replies, filter out any empty ones
    let chats = this.state.chats.filter((chat) => chat.msg !== "");
    let replies = this.state.agent_replies.filter((reply) => reply.msg !== "");
    chats = chats.filter((chat) => chat.msg);
    replies = replies.filter((reply) => reply.msg);
    // Label each chat based on where it came from
    chats.forEach((chat) => (chat["sender"] = "message user"));
    replies.forEach((reply) => (reply["sender"] = "message agent"));
    // Strip out the 'Agent: ' prefix if it's there
    replies.forEach(function (reply) {
      if (reply["msg"].includes("Agent: ")) {
        reply["msg"] = reply["msg"].substring(7);
      }
    });
    // Zip it into one list, sort by timestamp, and send it off to be rendered
    let chat_history = chats.concat(replies);
    chat_history.sort(function (a, b) {
      if (a.isQuestion && !b.isQuestion) {
        return 1;
      } else if (!a.isQuestion && b.isQuestion) {
        return -1;
      }
      return a.timestamp - b.timestamp;
    });

    return chat_history.map((chat) =>
      React.cloneElement(
        <li className="message-item" key={chat.timestamp.toString()}>
          <div className={chat.sender}>{chat.msg}</div>
          <div className="answer-buttons">
            <Button
              variant="contained"
              color="primary"
              className="yes-button"
              style={{ display: chat.isQuestion ? "inline flex" : "none" }}
              onClick={() => this.answerRouting(1, chat.questionType)}
            >
              Yes
            </Button>
            <Button
              variant="contained"
              color="primary"
              className="no-button"
              style={{ display: chat.isQuestion ? "inline flex" : "none" }}
              onClick={() => this.answerRouting(2, chat.questionType)}
            >
              No
            </Button>
            <Button
              variant="contained"
              color="primary"
              className="back-button"
              style={{ display: chat.enableBack ? "inline flex" : "none" }}
              onClick={() => this.answerRouting(3, chat.questionType)}
            >
              Go Back
            </Button>
          </div>
        </li>
      )
    );
  }

  handleKeyPress(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      this.handleSubmit();
    }
  }

  handleSubmit() {
    //get the message
    var chatmsg = document.getElementById("msg").value;
    //clear the textbox and previous questions
    document.getElementById("msg").value = "";
    this.removeButtonsFromLastQuestion();
    if (this.state.isSaveFeedback) {
      this.setState({
        isSaveFeedback: false,
        feedback: chatmsg,
      });
      this.saveFeedback();
      this.updateChat({ msg: chatmsg, timestamp: Date.now() });
      this.addNewAgentReplies({
        msg: "Feedback has been saved!" + PLEASE_RESUME,
      });
      this.removeButtonsFromLastQuestion();
    } else if (chatmsg.replace(/\s/g, "") !== "") {
      //add to chat history box of parent
      this.updateChat({ msg: chatmsg, timestamp: Date.now() });
      this.setState({ last_command: chatmsg });
      //log message to flask
      this.props.stateManager.logInteractiondata("text command", chatmsg);
      //log message to Mephisto
      window.parent.postMessage(
        JSON.stringify({ msg: { command: chatmsg } }),
        "*"
      );
      //send message
      this.props.stateManager.sendCommandToTurkInfo(chatmsg);
      this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
      // status updates
      this.props.stateManager.memory.commandState = "sent";
      if (this.state.agentType === "craftassist") {
        this.handleAgentThinking();
      }
    }
  }

  // Scroll to bottom when submit new message
  scrollToBottom = () => {
    if (this.messagesEnd) {
      this.messagesEnd.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };

  issueResetCommand() {
    if (this.state.clarify) {
      this.removeButtonsFromLastQuestion();
    }
    // update chat with "reset" instead of "stop" to avoid confusion
    this.updateChat({ msg: "reset", timestamp: Date.now() });
    // send message
    this.props.stateManager.logInteractiondata("text command", "stop");
    this.props.stateManager.socket.emit("sendCommandToAgent", "stop");
    this.props.stateManager.memory.commandState = "sent";
  }

  /**********************************************************************************
   ******************************* Agent Status Updates ******************************
   **********************************************************************************/

  handleAgentThinking() {
    if (this.props.stateManager) {
      this.props.stateManager.socket.on(
        "taskStackPollResponse",
        this.receiveTaskStackPoll
      );
    }

    this.intervalId = setInterval(() => {
      let commandState = null;

      if (this.props.stateManager) {
        commandState = this.props.stateManager.memory.commandState;
      }

      // Check that we're in an allowed state and haven't timed out
      if (this.safetyCheck()) {
        this.setState((prevState) => {
          if (prevState.commandState !== commandState) {
            // Log changes in command state to mephisto for analytics
            window.parent.postMessage(
              JSON.stringify({ msg: commandState }),
              "*"
            );
          }
          if (prevState.ellipsis.length > 6) {
            return {
              ellipsis: "",
              commandState: commandState,
            };
          } else {
            return {
              ellipsis: prevState.ellipsis + ".",
              commandState: commandState,
            };
          }
        });
      }
    }, this.props.stateManager.memory.commandPollTime);

    this.setState({
      commandState: this.props.stateManager.memory.commandState,
      now: Date.now(),
    });
  }

  componentDidUpdate(prevProps, prevState) {
    if (this.state.commandState !== prevState.commandState) {
      if (this.state.commandState !== "idle") {
        let disableInput = true;
        this.setState({
          disableInput: disableInput,
        });
      }
    }
    this.scrollToBottom();
  }

  renderResetButton() {
    // render during agent thinking and clarification
    if (this.state.commandState === "idle" && !this.state.clarify) {
      return;
    }

    return (
      <div className="reset">
        <Button
          variant="contained"
          color="primary"
          onClick={this.issueResetCommand.bind(this)}
          className="reset-button"
        >
          {this.state.clarify ? "Reset" : "Stop"}
        </Button>
      </div>
    );
  }

  renderStatusMessages() {
    if (this.state.commandState === "idle") {
      return;
    }

    let status_message = "";
    if (this.state.commandState === "sent") {
      status_message = "Sending command...";
    } else if (this.state.commandState === "received") {
      status_message = "Command received";
    } else if (this.state.commandState === "thinking") {
      status_message = "Assistant is thinking...";
    } else if (this.state.commandState === "done_thinking") {
      status_message = "Assistant is done thinking";
    } else if (this.state.commandState === "executing") {
      status_message = "Assistant is doing the task...";
    }

    return <div className="status">{status_message}</div>;
  }

  sendTaskStackPoll() {
    this.props.stateManager.socket.emit("taskStackPoll");
  }

  receiveTaskStackPoll(res) {
    console.log("Received task stack poll response:" + JSON.stringify(res));
    // If we get a response of any kind, reset the timeout clock
    if (res) {
      this.setState({
        now: Date.now(),
      });
      if (!res.task) {
        // If there's no task, leave this state
        if (this.state.isTurk) {
          this.askActionQuestion();
        }
        this.handleClearInterval();
      } else if (res.task && res.clarify) {
        console.log("Agent asked for task clarification");
        this.setState({ clarify: true });
        setTimeout(() => {
          this.sendTaskStackPoll();
        }, 1000);
      } else {
        // Otherwise send out a new task stack poll after a delay
        setTimeout(() => {
          this.sendTaskStackPoll();
        }, 1000);
      }
    }
  }

  safetyCheck() {
    // If we've gotten here during idle somehow, or timed out, escape to safety
    if (
      !this.allowedStates.includes(this.state.commandState) ||
      Date.now() - this.state.now > 50000
    ) {
      console.log("Safety dance: " + this.state.commandState);
      this.handleClearInterval();
      if (this.state.isTurk) {
        this.askActionQuestion();
      }
      return false;
    } else {
      return true;
    }
  }

  // Stop sending command
  handleClearInterval() {
    clearInterval(this.intervalId);
    if (this.props.stateManager) {
      this.props.stateManager.socket.off(
        "taskStackPollResponse",
        this.receiveTaskStackPoll
      );
      this.setState({
        disableInput: false,
        commandState: "idle",
        clarify: false,
      });
    }
  }

  /**********************************************************************************
   ********************************* Error Marking ***********************************
   ***********************************************************************************/

  answerRouting(index, questionType) {
    switch (questionType) {
      case ANSWER_ACTION:
        this.answerAction(index);
        break;
      case ANSWER_PARSING:
        this.answerParsing(index);
        break;
      case ANSWER_VISION:
        this.answerVision(index);
        break;
      case CLARIFICATION:
        this.answerClarification(index);
        break;
      default:
        console.error("Answer Routing called with invalid question type!");
    }
  }

  askActionQuestion() {
    // Send request to retrieve the logic form of last sent command
    this.props.stateManager.socket.emit(
      "getChatActionDict",
      this.state.last_command
    );

    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(
      JSON.stringify({ msg: "askActionQuestion" }),
      "*"
    );

    this.addNewAgentReplies({
      msg: "Did I successfully do the task you asked me to complete?",
      isQuestion: true,
      questionType: ANSWER_ACTION,
      enableBack: false,
    });
  }

  answerAction(index) {
    if (index === 1) {
      // Yes, so no error
      this.updateChat({ msg: "Yes", timestamp: Date.now() });
      this.addNewAgentReplies({
        msg: "Thanks!" + PLEASE_RESUME,
        questionType: ANSWER_PARSING,
        isQuestion: false,
        disablePreviousAnswer: true,
        enableBack: true,
      });
      this.setState({ disableInput: false });
    } else if (index === 2) {
      this.setState({ task_error: true });
      // No, there was an error of some kind
      if (this.state.action_dict) {
        this.updateChat({ msg: "No", timestamp: Date.now() });
        if ("dialogue_type" in this.state.action_dict) {
          var dialogue_type = this.state.action_dict.dialogue_type;
          var question_word = "";
          if (dialogue_type === "HUMAN_GIVE_COMMAND") {
            // handle composite action

            // get the action type
            var action_dict = this.state.action_dict.event_sequence[0];
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
                else if (
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
        this.addNewAgentReplies({
          msg: `Did you want the assistant ${question_word}`,
          isQuestion: true,
          questionType: ANSWER_PARSING,
          disablePreviousAnswer: true,
          enableBack: true,
        });
      } else {
        // shouldn't happen
        this.updateChat({ msg: "No", timestamp: Date.now() });
        this.addNewAgentReplies({
          msg: "Thanks!" + PLEASE_RESUME,
          isQuestion: false,
          questionType: ANSWER_PARSING,
          disablePreviousAnswer: true,
          enableBack: true,
        });
      }
    }
  }

  //handles after the user submits the answer (y/n) to if NSP errored or not
  answerParsing(index) {
    this.removeButtonsFromLastQuestion();
    if (index === 1) {
      // yes, so not a parsing error
      this.updateChat({ msg: "Yes", timestamp: Date.now() });
      this.evalCommandPerception();
      this.askVisionQuestion();
    } else if (index === 2) {
      // no, so parsing error
      this.updateChat({ msg: "No", timestamp: Date.now() });
      this.renderParsingFail();
      this.setState({ parsing_error: true });
    } else if (index === 3) {
      // go back to the beginning
      this.updateChat({ msg: "Go Back", timestamp: Date.now() });
      this.setState({
        parsing_error: false,
        vision_error: false,
        task_error: false,
        disableInput: true,
        isSaveFeedback: false,
      });
      this.askActionQuestion();
    }
  }

  renderParsingFail() {
    this.addNewAgentReplies({
      msg:
        "Thanks for letting me know that I didn't understand the command right." +
        PLEASE_RESUME,
      questionType: ANSWER_PARSING,
      disablePreviousAnswer: true,
      enableBack: false,
    });
    this.saveFeedback();
  }

  check_reference_object_in_action_dict(action) {
    var action_dict = action;
    for (var key in action_dict) {
      if (key === "reference_object") {
        return true;
      } else {
        if (action_dict[key].constructor === Object) {
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
      if (this.state.action_dict["dialogue_type"] === "HUMAN_GIVE_COMMAND") {
        // also implement for get and put memory
        for (const action of this.state.action_dict.event_sequence) {
          ref_object = this.check_reference_object_in_action_dict(action);
        }
      }

      // If yes, find reference object description.
      if (ref_object === true) {
        const action_dict = this.state.action_dict.event_sequence[0];
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
      // If no reference object description found not a perception error.
    }
    if (!reference_object_description) {
      console.log(
        "InteractApp evalCommandPerception: no action dictionary found"
      ); // Shouldn't happen....
    }
    this.setState({
      reference_object_description: reference_object_description,
    });
  }

  askVisionQuestion() {
    //        Is 'This' the Location Ref?
    //                  /   \
    // Other Error <- Yes    No -> Perception Error
    let reference_object_description = this.state.reference_object_description;
    if (!reference_object_description) {
      // Not perception error.
      this.renderOtherError();
      return;
    }

    this.setState({
      agent_replies: this.props.stateManager.memory.agent_replies,
    });
    if (this.state.memory_entries) {
      var point_target = this.state.memory_entries["point_target"];
      var bbox = point_target.join(" ");
      console.log("bbox: " + bbox);
      this.props.stateManager.flashBlocksInVW(bbox);
      this.addNewAgentReplies({
        msg: `Okay, I was looking for an object of interest called :
          ${reference_object_description}. I'll make it flash in the world now.
          Does that object look right ?`,
        isQuestion: true,
        questionType: ANSWER_VISION,
        enableBack: true,
      });
    } else {
      this.addNewAgentReplies({
        msg: `I did not find anything called : ${reference_object_description} in the world.
          Does that seem right from your view of the world ?`,
        isQuestion: true,
        questionType: ANSWER_VISION,
        enableBack: true,
      });
    }
  }

  answerVision(index) {
    //handles after the user submits the answer (y/n) to if NSP errored or not
    if (index === 1) {
      // yes, so not a vision error
      this.updateChat({ msg: "Yes", timestamp: Date.now() });
      this.renderOtherError();
    } else if (index === 2) {
      // no, so vision error
      this.updateChat({ msg: "No", timestamp: Date.now() });
      this.setState({ vision_error: true });
      this.renderVisionFail();
    } else if (index === 3) {
      // go back to parsing question
      this.updateChat({ msg: "Go Back", timestamp: Date.now() });
      this.setState({
        disableInput: true,
        isSaveFeedback: false,
        parsing_error: false,
        vision_error: false,
      });
      this.answerAction(2);
    }
  }

  renderVisionFail() {
    this.addNewAgentReplies({
      msg:
        "Thanks for letting me know that I didn't detect the object right." +
        PLEASE_RESUME,
      questionType: ANSWER_VISION,
      disablePreviousAnswer: true,
      enableBack: false,
    });
    this.saveFeedback();
  }

  renderOtherError() {
    this.addNewAgentReplies({
      msg: "Okay, looks like I understood your command but didn't complete it. Please tell me more about what I did wrong?",
      questionType: ANSWER_VISION,
      isQuestion: false,
      enableBack: true,
    });
    this.setState({
      disableInput: false,
      isSaveFeedback: true,
    });
  }

  /**********************************************************************************
   ********************************** Clarification **********************************
   **********************************************************************************/

  answerClarification(index) {
    //handles answer to clarification question
    let chatmsg;
    if (index === 1) {
      chatmsg = "yes";
    } else if (index === 2) {
      chatmsg = "no";
    }
    this.updateChat({ msg: chatmsg, timestamp: Date.now() });
    this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
    this.removeButtonsFromLastQuestion();
  }

  /**********************************************************************************
   ************************************* Render **************************************
   **********************************************************************************/

  render() {
    return (
      <div className="App" style={{ padding: 0 }}>
        <div className="content">
          <div>
            <div>
              <p>
                Enter the command to the assistant in the input box below, then
                press 'Enter'.
              </p>
            </div>
            <div className="center">
              <div className="chat">
                <div className="time">
                  Assistant is{" "}
                  {this.state.connected === true ? (
                    <span style={{ color: "green" }}>connected</span>
                  ) : (
                    <span style={{ color: "red" }}>not connected</span>
                  )}
                </div>
                <div className="messages">
                  <div className="messsages-content" id="scrollbar">
                    <ul className="messagelist" id="chat">
                      {this.renderChatHistory()}
                      <div
                        className="messagesEnd"
                        ref={(el) => {
                          this.messagesEnd = el;
                        }}
                      ></div>
                    </ul>
                  </div>
                </div>
                {this.renderResetButton()}
                {this.renderStatusMessages()}
                <div className="input">
                  <input
                    id="msg"
                    placeholder={
                      this.state.disableInput
                        ? `Waiting for Assistant${this.state.ellipsis}`
                        : "Type your command or response here"
                    }
                    type="text"
                    disabled={this.state.disableInput}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default InteractApp;
