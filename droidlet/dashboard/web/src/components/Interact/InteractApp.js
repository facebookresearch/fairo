import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import "./InteractApp.css";

const ANSWER_ACTION = "answerAction";
const ANSWER_PARSING = "answerParsing";
const ANSWER_VISION = "answerVision";
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
      disableStopButton: true,
      lastChatActionDict: "",
      chats: [{ msg: "", timestamp: Date.now() }],
      agent_replies: [{}],
      agentType: null,
      isTurk: false,
      action_dict: {},
      parsing_error: false,
      perception_error: false,
      task_error: false,
      feedback: "",
      isSaveFeedback: false,
    };

    this.state = this.initialState;
    this.elementRef = React.createRef();
    this.bindKeyPress = this.handleKeyPress.bind(this); // this is used in keypressed event handling
    this.sendTaskStackPoll = this.sendTaskStackPoll.bind(this);
    this.receiveTaskStackPoll = this.receiveTaskStackPoll.bind(this);
    this.issueStopCommand = this.issueStopCommand.bind(this);
    this.handleAgentThinking = this.handleAgentThinking.bind(this);
    this.handleClearInterval = this.handleClearInterval.bind(this);
    this.askActionQuestion = this.askActionQuestion.bind(this);
    this.intervalId = null;
    this.messagesEnd = null;
    this.addNewAgentReplies = this.addNewAgentReplies.bind(this);
    this.answerActionYes = this.answerActionYes.bind(this);
    this.answerActionNo = this.answerActionNo.bind(this);
    this.answerParsing = this.answerParsing.bind(this);
    this.evalCommandPerception = this.evalCommandPerception.bind(this);
    this.askVisionQuestion = this.askVisionQuestion.bind(this);
    this.renderOtherError = this.renderOtherError.bind(this);
    this.disableAnswer = this.disableAnswer.bind(this);
    this.answerVision = this.answerVision.bind(this);
    this.renderVisionFail = this.renderVisionFail.bind(this);
    this.renderParsingFail = this.renderParsingFail.bind(this);
    this.saveFeedback = this.saveFeedback.bind(this);
    this.removeButtonsFromLastQuestion =
      this.removeButtonsFromLastQuestion.bind(this);
  }

  saveFeedback(event) {
    //save feedback in state
    this.setState({ feedback: event });
  }

  renderOtherError() {
    this.addNewAgentReplies({
      msg: "Okay, looks like I understood your command but didn't complete it. Please tell me more about what I did wrong?",
      isQuestion: false,
    });
    this.setState({
      disableInput: false,
      agent_replies: this.props.stateManager.memory.agent_replies,
      disableStopButton: true,
      isSaveFeedback: true,
    });
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
      // If no reference object description found not a perception error.
    }
    if (!reference_object_description) {
      console.log(
        "InteractApp evalCommandPerception: no action dictionary found"
      ); // Shouldn't happen....
    }
    const self = this;
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
    // Check for this reference object in memory
    let user_message = null;
    // NOTE: this should come from the state setter sio event.
    this.state.memory_entries = null;
    if (this.state.memory_entries) {
      this.addNewAgentReplies({
        msg: `Okay, I was looking for an object of interest called :
          ${reference_object_description}. I'll make it flash in the world now.
          Does that object look right ?`,
        isQuestion: true,
        questionType: ANSWER_VISION,
      });
    } else {
      this.addNewAgentReplies({
        msg: `I did not find anything called : ${reference_object_description} in the world.
          Does that seem right from your view of the world ?`,
        isQuestion: true,
        questionType: ANSWER_VISION,
      });
    }
  }

  removeButtonsFromLastQuestion() {
    console.log(
      "InteractApp removeButtonsFromLastQuestion " +
        JSON.stringify(this.state.chats)
    );
    var new_agent_replies = [...this.state.agent_replies];
    new_agent_replies.map((agent_reply) => (agent_reply.isQuestion = false));
    this.setState({ agent_replies: new_agent_replies });
  }

  //handles after the user submits the answer (y/n) to if NSP errored or not
  answerParsing(index) {
    this.removeButtonsFromLastQuestion();
    if (index === 1) {
      // yes, so not a parsing error
      this.updateChat({ msg: "Yes", timestamp: Date.now() });
      this.evalCommandPerception();
      // this.setState({ view: 3 });
      this.askVisionQuestion();
    } else if (index === 2) {
      // no, so parsing error
      this.updateChat({ msg: "No", timestamp: Date.now() });
      this.renderParsingFail();
      this.setState({ parsing_error: true });
    }
  }

  renderParsingFail() {
    this.removeButtonsFromLastQuestion();
    this.addNewAgentReplies({
      msg:
        "Thanks for letting me know that I didn't understand the command right." +
        PLEASE_RESUME,
    });
  }

  renderVisionFail() {
    this.removeButtonsFromLastQuestion();
    this.addNewAgentReplies({
      msg:
        "Thanks for letting me know that I didn't detect the object right." +
        PLEASE_RESUME,
    });
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
      this.renderVisionFail();
      this.setState({ vision_error: true });
    }
  }

  setAnswerIndex(index) {
    this.setState({
      answerIndex: index,
    });
  }

  updateChat(chat) {
    console.log("InteractApp updateChat: " + JSON.stringify(chat));
    // make a shallow copy of chats
    var new_chats = [...this.state.chats];
    new_chats.push(chat);
    this.setState({ chats: new_chats });
  }

  sendTaskStackPoll() {
    //console.log("Sending task stack poll");
    this.props.stateManager.socket.emit("taskStackPoll");
  }

  receiveTaskStackPoll(res) {
    var response = JSON.stringify(res);
    // console.log("Received task stack poll response:" + response);
    // If we get a response of any kind, reset the timeout clock
    // console.log(res);
    if (res) {
      this.setState({
        now: Date.now(),
      });
      if (!res.task) {
        console.log("InteractApp: no task on stack");
        // If there's no task, leave this pane
        // If it's a HIT go to error labeling, else back to Message
        if (this.state.isTurk) {
          this.askActionQuestion(this.state.chats.length - 1);
        }
        this.handleClearInterval();
      } else {
        // Otherwise send out a new task stack poll after a delay
        setTimeout(() => {
          this.sendTaskStackPoll();
        }, 1000);
      }
    }
  }

  issueStopCommand() {
    console.log("Stop command issued");
    const chatmsg = "stop";
    //add to chat history box of parent
    this.updateChat({ msg: chatmsg, timestamp: Date.now() });
    //log message to flask
    this.props.stateManager.logInteractiondata("text command", chatmsg);
    //socket connection
    this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
    //update StateManager command state
    this.props.stateManager.memory.commandState = "sent";
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
    // console.log("InteractApp renderChatHistory: " + JSON.stringify(chat_history));
    chat_history.sort(function (a, b) {
      if (a.isQuestion && !b.isQuestion) {
        return 1;
      } else if (!a.isQuestion && b.isQuestion) {
        return -1;
      } else if (!a.isQuestion && b.isQuestion) {
        console.log("InteractApp renderChatHistory -- there are two questions");
      }
      return a.timestamp - b.timestamp;
    });

    return chat_history.map((chat) =>
      React.cloneElement(
        <li className="message-item" key={chat.timestamp.toString()}>
          <div className={chat.sender}>{chat.msg}</div>
          {chat.isQuestion && chat.questionType === ANSWER_ACTION && (
            <div className="answer-buttons">
              <Button
                variant="contained"
                color="primary"
                className="yes-button"
                onClick={() => this.answerActionYes()}
              >
                Yes
              </Button>
              <Button
                variant="contained"
                color="primary"
                className="no-button"
                onClick={() => this.answerActionNo()}
              >
                No
              </Button>
            </div>
          )}
          {chat.isQuestion && chat.questionType === ANSWER_PARSING && (
            <div className="answer-buttons">
              <Button
                variant="contained"
                color="primary"
                className="yes-button"
                onClick={() => this.answerParsing(1)}
              >
                Yes
              </Button>
              <Button
                variant="contained"
                color="primary"
                className="no-button"
                onClick={() => this.answerParsing(2)}
              >
                No
              </Button>
            </div>
          )}
          {chat.isQuestion && chat.questionType === ANSWER_VISION && (
            <div className="answer-buttons">
              <Button
                variant="contained"
                color="primary"
                className="yes-button"
                onClick={() => this.answerVision(1)}
              >
                Yes
              </Button>
              <Button
                variant="contained"
                color="primary"
                className="no-button"
                onClick={() => this.answerVision(2)}
              >
                No
              </Button>
            </div>
          )}
        </li>
      )
    );
  }

  disableAnswer() {
    console.log("InteractApp disableAnswer");
    const new_agent_replies = this.state.agent_replies.map((item) => ({
      ...item,
      isQuestion: false,
    }));
    this.setState({
      agent_replies: new_agent_replies,
    });
  }

  answerActionYes() {
    this.updateChat({ msg: "Yes", timestamp: Date.now() });
    this.addNewAgentReplies({
      msg: "Thanks!" + PLEASE_RESUME,
      isQuestion: false,
      disablePreviousAnswer: true,
    });
  }

  answerActionNo() {
    console.log("actionAnswerNo " + JSON.stringify(this.state.action_dict));
    if (this.state.action_dict) {
      this.updateChat({ msg: "No", timestamp: Date.now() });
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
      this.addNewAgentReplies({
        msg: `Did you want the assistant ${question_word}`,
        isQuestion: true,
        questionType: ANSWER_PARSING,
        disablePreviousAnswer: true,
      });
    } else {
      // shouldn't happen
      this.updateChat({ msg: "No", timestamp: Date.now() });
      this.addNewAgentReplies({
        msg: "Thanks!" + PLEASE_RESUME,
        isQuestion: false,
        disablePreviousAnswer: true,
      });
    }
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  handleKeyPress(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      this.handleSubmit();
    }
  }

  componentDidMount() {
    console.log("InteractApp componentDidMount");
    document.addEventListener("keypress", this.bindKeyPress);
    if (this.props.stateManager) {
      this.props.stateManager.connect(this);
      var lastChatActionDict =
        this.props.stateManager.memory.lastChatActionDict;
      this.setState({
        isTurk: this.props.stateManager.memory.isTurk,
        agent_replies: this.props.stateManager.memory.agent_replies,
        connected: this.props.stateManager.connected,
        action_dict: lastChatActionDict,
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

  askActionQuestion(idx) {
    // Send a message to the parent iframe for analytics logging
    window.parent.postMessage(
      JSON.stringify({ msg: "askActionQuestion" }),
      "*"
    );

    const chats_len = this.state.chats.length;

    this.addNewAgentReplies({
      msg: "Did I successfully do the task you asked me to complete?",
      isQuestion: true,
      questionType: ANSWER_ACTION,
    });

    this.setState({
      agent_replies: this.props.stateManager.memory.agent_replies,
      chats: this.state.chats,
    });

    // Send request to retrieve the logic form of last sent command
    this.props.stateManager.socket.emit(
      "getChatActionDict",
      this.state.chats[idx]["msg"]
    );
  }

  handleSubmit() {
    //get the message
    var chatmsg = document.getElementById("msg").value;
    if (this.state.isSaveFeedback) {
      this.saveFeedback(chatmsg);
      document.getElementById("msg").value = "";
      this.updateChat({ msg: chatmsg, timestamp: Date.now() });
      console.log("InteractApp save feedback: " + chatmsg);
      this.addNewAgentReplies({
        msg: "Feedback has been saved!" + PLEASE_RESUME,
      });
      this.removeButtonsFromLastQuestion();
      this.setState({
        isSaveFeedback: false,
      });
    } else {
      if (chatmsg.replace(/\s/g, "") !== "") {
        //add to chat history box of parent
        this.updateChat({ msg: chatmsg, timestamp: Date.now() });
        //log message to flask
        this.props.stateManager.logInteractiondata("text command", chatmsg);
        //log message to Mephisto
        window.parent.postMessage(
          JSON.stringify({ msg: { command: chatmsg } }),
          "*"
        );
        //send message to TurkInfo
        this.props.stateManager.sendCommandToTurkInfo(chatmsg);
        //socket connection
        this.props.stateManager.socket.emit("sendCommandToAgent", chatmsg);
        //update StateManager command state
        this.props.stateManager.memory.commandState = "sent";
        //clear the textbox
        document.getElementById("msg").value = "";
        //clear the agent reply that will be shown in the question pane
        this.props.stateManager.memory.last_reply = "";
        //execute agent thinking function if it makes sense
        if (this.state.agentType === "craftassist") {
          this.handleAgentThinking();
        }
      }
    }
  }

  // Merge agent thinking functionality
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
        //console.log("Command State from agent thinking: " + commandState);
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

  safetyCheck() {
    // If we've gotten here during idle somehow, or timed out, escape to safety
    if (
      !this.allowedStates.includes(this.state.commandState) ||
      Date.now() - this.state.now > 30000
    ) {
      console.log("Safety dance: " + this.state.commandState);
      this.askActionQuestion(this.state.chats.length - 1);
      this.handleClearInterval();
      return false;
    } else {
      return true;
    }
  }

  // Stop sending command
  handleClearInterval() {
    console.log("InteractApp handleClearInterval");
    clearInterval(this.intervalId);
    if (this.props.stateManager) {
      this.props.stateManager.socket.off(
        "taskStackPollResponse",
        this.receiveTaskStackPoll
      );
      this.setState({
        disableInput: false,
        disableStopButton: true,
      });
    }
  }

  // Scroll to bottom when submit new message
  scrollToBottom = () => {
    if (this.messagesEnd)
      this.messagesEnd.scrollIntoView({ behavior: "smooth" });
  };

  componentDidUpdate(prevProps, prevState) {
    // Show command message like an agent reply
    if (this.state.commandState !== prevState.commandState) {
      console.log(
        "InteractApp componentDidUpdate command_state: " +
          this.state.commandState
      );
      let command_message = "";
      let disableInput = true;
      let disableStopButton = this.state.disableStopButton;
      if (this.state.commandState === "sent") {
        command_message = "Sending command...";
        disableStopButton = true;
      } else if (this.state.commandState === "received") {
        command_message = "Command received";
        disableStopButton = true;
      } else if (this.state.commandState === "thinking") {
        command_message = "Assistant is thinking...";
        disableStopButton = true;
      } else if (this.state.commandState === "done_thinking") {
        command_message = "Assistant is done thinking";
        disableStopButton = false;
      } else if (this.state.commandState === "executing") {
        command_message = "Assistant is doing the task";
        disableStopButton = false;
      }
      if (command_message) {
        console.log(
          "InteractApp componentDidUpdate command_message: " + command_message
        );
        const new_agent_replies = [
          ...this.state.agent_replies,
          { msg: command_message, timestamp: Date.now() },
        ];
        this.setState({
          agent_replies: new_agent_replies,
          disableInput: disableInput,
          disableStopButton: disableStopButton,
        });
      }
    }
    // Scroll messsage panel to bottom
    this.scrollToBottom();
  }

  addNewAgentReplies({ msg, isQuestion, questionType, disablePreviousAnswer }) {
    console.log("InteractApp addNewAgentReplies " + msg);
    const { agent_replies } = this.state;
    let new_agent_replies = disablePreviousAnswer
      ? agent_replies.map((item) => ({ ...item, isQuestion: false }))
      : agent_replies;
    new_agent_replies = [
      ...new_agent_replies,
      {
        msg: msg,
        timestamp: Date.now() + 1,
        questionType: questionType,
        isQuestion: isQuestion,
      },
    ];
    this.setState({
      agent_replies: new_agent_replies,
    });
    this.props.stateManager.memory.agent_replies = new_agent_replies;
  }

  render() {
    //    console.log(this.props.stateManager.memory.agent_replies);
    //    console.log(this.props.stateManager);
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
                    </ul>
                    <div
                      style={{ float: "left", clear: "both" }}
                      ref={(el) => {
                        this.messagesEnd = el;
                      }}
                    ></div>
                  </div>
                </div>
                <div className="input">
                  <input
                    id="msg"
                    placeholder={
                      this.state.disableInput
                        ? `Waiting for Assistant${this.state.ellipsis}`
                        : "Type your command here"
                    }
                    type="text"
                    disabled={this.state.disableInput}
                  />
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={this.issueStopCommand.bind(this)}
                    className="stop-button"
                    disabled={this.state.disableStopButton}
                  >
                    Stop
                  </Button>
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
