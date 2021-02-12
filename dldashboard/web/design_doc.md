# Design document for dashboard

## Architecture 
A high level design diagram of the current architecture of the dashboard is :

![dashboard architecture diagram](https://craftassist.s3-us-west-2.amazonaws.com/pubr/dashboard_architecture.png)

In the diagram above:
- `**frontend**`: The React based frontend that hosts other React subcomponents in a given layout. All the components are expected to reflect the most up-to-date state whenever there is a state change communicated from backend. 
- `**StateManager**`: The module that manages messaging between `frontend` and `backend`. Through the StateManager : we set the state of the React components with the data communicated from backend, and the components can explicitly request information / set the state of backend. The `StateManager` and the `backend` communicate wiht each other using `socket.io` connections. The goal is to have `StateManager` be completely in sync with the backend but we aren't there yet.
- `**backend**` : This is the agent process that sends and receives data from the `frontend` through the `StateManager` component. The agent process and its modules (perception, memory, mover, controller etc) can communicate with `StateManager` using `socket.io` connections. 


## Snapshot 
Here's a snapshot of how the current dashboard looks like :
![dashboard screenshot](https://craftassist.s3-us-west-2.amazonaws.com/pubr/dashboard_screenshot.png)

On a high level, following is the layout in the source code :
- `Index.js` is the main React component that hosts other React Components in the following layout :
  - Left:
    - `MainPane.js`:
      - Top:
        - `InteractApp`: Lets you send a command to the assistant using voice or text. The commands once sent, are clickable to annotate if there was a language understanding error. This component also shows the status of the command execution by displaying agent reply or agent error messages.
      - Bottom:
        - `LiveImage` - Shows the rgb image stream
        - `LiveImage` - Shows the depth map
        - `LiveObjects` - Shows detector output
        - `LiveHumans` - Shows human pose estimation
  - Right:
    - Top right:
      - `Memory2D`: Visualizes the memory map on a 2D grid.
      - `History`: Shows the past 5 chat interactions through the dashboard.
      - Query the Semantic Parser (`QuerySemanticParser`): Lets you query the semantic parser with some text.
      - Program The Assistant (`TeachApp`): Lets you do construct new programs for the assistant by assembling together blockly like blocks.
    - Bottom right:
      - `Settings`: Shows current settings, connection status, image quality, resolution etc
      - `Navigator`: Lets you navigate the assistant through keypress (up, down, left, right)
      - `ObjectFixUp`: Lets you annotate objects in dashboard.


## Dashboard components 

The following table shows the frontend components, the socket events they emit/handle and explains the purpose of those events :


| Component Name | Socket Event | Purpose | Socket event sent by |
| ------------- | ------------- | ------------- | ------------- |
| InteractApp | saveErrorDetailsToDb | To save the error category to backend db | frontend |
| InteractApp  | sendCommandToAgent  | send command from frontend to agent process | frontend |
| InteractApp | setChatResponse | Get parsing status, logical form for this chat from semantic parser, history of past 5 commands | backend |
| InteractApp | showAssistantreply | Get the agent's reply after processing the command, if any and render it.  | backend |
| LiveImage | sensor_payload | Get rgb image stream and depth information from backend | backend |
| LiveObjects | senor_payload | Get objects and their rgb information from detector on backend | backend |
| LiveHumans | sensor_payload | Get human pose estimation from backend | backend |
| Memory2D | sensor_payload | Get objects, x, y and yaw of assistant and obstacle map from backend and render it on a 2D grid | backend |
| History | setChatResponse | Render the past 5 commands that were sent to the assistant through the dashboard | backend |
| QuerySemanticParser | queryParser | send a command to query the semantic parser | frontend |
| QuerySemanticParser | renderActionDict | Render the parser output for the command sent above | backend |
| TeachApp | saveCommand | Save the command the user just defined using the blocks | frontend |
| TeachApp | fetchCommand | Fetch the blocks for the input command and the action dictionary | frontend |
| TeachApp | updateSearchList | Render the search list with partially input command | backend |
| Navigator | command | Lets you navigate the assistant through keypress (up, down, left, right) | frontend |
| ObjectFixUp | saveObjectAnnotation | Lets you annotate mask and properties of objects in dashboard and save them to backend | frontend |
