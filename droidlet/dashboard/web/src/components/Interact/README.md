## Chat Architecture Flowchart ##
<pre>
<i>handleSubmit</i>                              User sends a new chat (command status: 'sent')
                                                      |
<i>setChatReponse</i> (in <i>StateManager</i>)         Agent receives the chat (command status: 'received' -> 'thinking' (500ms later))
                                                      |
<i>returnTimelineEvent</i> (in <i>StateManager</i>)     Agent parses the chat (command status: 'done_thinking')
                      ______________________/_________|_____   \
                      |                    /          |     |   \ 
<i>handleAgentThinking</i>   | Clarification needed          |      \--Task added to stack (command status: 'executing')
                      |           |           Uncaught failure           |       \
<i>answerClarification</i>   |     Verify parse              |                  |      "Stop" command issued  <i>issueResetCommand</i>
                      |          y|     \n_____       |            Task Complete          |
<i>answerClarification</i>   |   Clarify ref objs     \      |       __________/      __________/
                      |_y/              \n____  \     |      /   _____________/
                                              \  \    |     /   /
                                              Error Marking Flow  (if Turk, otherwise end)
                                                      |
<i>askActionQuestion</i>                       Did the agent execute correctly?
                                      n/                               \y
<i>answerAction</i>                 Is the parse correct?             (finished - no error)
<i>answerParsing</i>              n/                     \y
                (finished - nlu error)   Was there a vision error?   <i>askVisionQuestion</i>
<i>answerVision</i>                           n/                         \y
                            (finished - other error)     (finished - vision error)
</pre>

## InteractApp Overview ##

`InteractApp.js` holds the chat interface for the dashboard, which includes sending and displaying chats to and from the agent, displaying the agent status updates while commands are being processed, providing a stop/reset button at the relevant moments, and displaying response buttons when limited response options are specified (eg. during error marking or clarification).  As there is a substantial amount of code in this component, below are descriptions of the purpose of each section.  In general, functions appear in the order in which they are called, though this is a rough guide at best.

### Component Utils ###

This short section near the top of the file contains utility functions that are used in multiple places throughout the app or are only called once, eg. `componentWillMount`.  `saveFeedback` is here because it is called from both error marking and messaging.

### Messaging ###

This section holds the functionality related to sending and displaying chats.  User chats come from the chat app itself are are stored in `this.state.chats`, while chats from the agent arrive via an sio event to the `StateManager`.  `addNewAgentReplies` pulls the agent chats in from the state manager and stores them in `this.state.agent_replies`.  To render the chat history in the window, the two lists are zipped based on timestamp, and any response option buttons are rendered below the chat if called for.

### Agent Status Updates ###

While the agent is idle, the agent status in `InteractApp` is "idle".  After submitting a command, the status updates pass through five states: 'sent', 'received', 'thinking', 'done_thinking', and 'executing'.  These messages are displayed in a thin tooltip directly above the chat input box.  The triggers for moving through these states vary - some are time based and others are triggered based on sio events.  The dashboard needs to poll the agent to learn when task execution is complete (`sendTaskStackPoll`).  Every second the dashboard sends a poll to the agent, who checks to see if there is still a task on the stack, and responds with the answer.  If there is, then the command is still being processed.  If there isn't, the dashboard can move on to error marking.  While in this state, the user is prevented from sending new commands, with the exception of a "stop" command, which they can send by hitting a stop button.  There is also a 50 second safety timeout implemented here, so that if the agent is working on a task longer than that amount of time, the dashboard will assume that there is an error and allow the user to submit another command.  This has the downside of disabling error marking on commands that take longer than this amount of time to complete.

### Error Marking ###

After the agent has finished executing each command, the user is prompted to mark whether the agent was successful before moving onto the next command.  The questions are presented to the user in the form of a decision tree, starting with the question "Did I successfully do the task you asked me to complete?"  If yes, then error marking is done and the user can submit the next command, if no then we attempt to find out what kind of error occurred (credit assignment).  First we ask if it was a parsing error, then a perception error or other error.  Each of these questions has to be constructed from the `action_dict`, which is the logical form produced by the NLU module in the agent.  The answer to each question, or pressing the 'go back' button, determines where in the decision tree the user progresses.

### Clarification ###

The clarification section contains functions that are specific to the clarification pathway in the agent.  This works differently because it is the only time when we have restricted response options that are also chats sent to the agent.  The chat history also needs to be remembered and formatted in a specific way.

### Render ###

The final section is the requisite `render()` call that all React components must have.  It includes all of the content mentioned above, as well as an indicator that displays whether or not the agent back end is connected to the dashboard.




