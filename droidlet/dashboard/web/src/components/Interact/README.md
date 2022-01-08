# Interaction with humans

We have the following files in this folder:

- Index.js
- AgentThinking.js
- InteractApp.js
  This is the main pane that hosts either `Message.js`, `Question.js` or `AgentThinking.js` in it.
- Labeling.js
- Message.js
  This pane has the text box through which humans can send a command to the agent, click on the command if there was an
  error. This component also shows the history of chat messages and whether or not the message
- was successfully sent to the backend agent process.
- `Question.js` :
  This pane walks through the sub-component credit assignment when the assistant makes a mistake, more on this in the next section.

## Credit assignment on failures

When the human marks that the assistant made a mistake, we first:

1. Ask if the action was executed properly (`renderActionQuestion`), if the answer to this is yes, we don't explore the
   following routes at all. The assistant made no mistake.
2. Show the command the human sent to the assistant and the reply the assistant sent back.
   We read the NLU output and translate it back to plain text using heuristics (`renderSemanticParserErrorQuestion`).
   For example for `"move to the blue house"`
   we ask: `"Did you want the assistant to move to location 'blue house'" ?` This then has two routes:
   1. Yes that is correct - (`renderParsingSuccess`)
   2. No that's not right - (`)
