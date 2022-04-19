# Design document for dashboard

## Architecture 
A high level design diagram of the current architecture of the dashboard is :

![dashboard architecture diagram](https://craftassist.s3-us-west-2.amazonaws.com/pubr/dashboard_architecture.png)

In the diagram above:
- **`frontend`**: The React based frontend that hosts other React subcomponents in a given layout. All the components are expected to reflect the most up-to-date state whenever there is a state change communicated from backend. 
- **`StateManager`**: The module that manages messaging between `frontend` and `backend`. Through the `StateManager`, we set the state of the React components with the data communicated from backend, and the components can explicitly request information / set the state of backend. The `StateManager` and the `backend` communicate with each other using `socket.io` connections. The goal is to have `StateManager` be completely in sync with the backend but we aren't there yet.
- **`backend`** : This is the agent process that sends and receives data from the `frontend` through the `StateManager`. The agent process and its modules (perception, memory, mover, controller etc) can also communicate with `StateManager` using `socket.io` connections. 


## Snapshot 
Here's a snapshot of how the current dashboard looks like :
![dashboard screenshot](https://craftassist.s3-us-west-2.amazonaws.com/pubr/dashboard_screenshot-2022.png)

On a high level, following is the layout in the source code :
- `Index.js` is the main React component that hosts other React Components in the following layout :
  - Left:
    - `MainPane.js`:
      - Top:
        - `InteractApp`: Lets you send a command to the assistant using text, and shows replies from the agent.  After each command, status messages will show the state of command processing.  After each command the user must label whether it was understood and carried out successfully before issuing the next command.
      - Bottom (depends on what agent back end is running):
        - If agent_type is craftassist:
          - `VoxelWorld` - Shows a randering of the environment
        - If agent_type is locobot:
          - `LiveImage` - Shows the rgb image stream
          - `LiveImage` - Shows the depth map
          - `LiveObjects` - Shows detector output
          - `LiveHumans` - Shows human pose estimation
  - Right:
    - Top right:
      - `Memory2D`: Visualizes the memory map on a 2D grid.
      - `MemoryList`: View the state of agent memory as a list
      - Query the Semantic Parser (`QuerySemanticParser`): Lets you query the semantic parser with some text.
      - Program The Assistant (`TeachApp`): Lets you do construct new programs for the assistant by assembling together blockly like blocks.
      - `Timeline`: A timeline of agent command task events
    - Bottom right:
      - `Settings`: Shows current settings, connection status, image quality, resolution etc
      - `Navigator`: Lets you navigate the assistant through keypress (up, down, left, right)
      - `ObjectFixUp`: Lets you annotate objects in dashboard.


## Dashboard components 

The following table shows the frontend components, the socket events they emit/handle and explains the purpose of those events :


| Component Name | Socket Event | Purpose | Socket event sent by |
| ------------- | ------------- | ------------- | ------------- |
| MainPane & InteractApp | updateAgentType | Show the correct panes based on agent type | backend |
| InteractApp | setLastChatActionDict | Display the logical form of last chat | backend |
| InteractApp | getChatActionDict | Send request to retrieve the logic form of last sent command | frontend |
| InteractApp | saveErrorDetailsToCSV | To save the error to backend db | frontend |
| InteractApp | taskStackPoll | Poll task stack to see if complete | frontend |
| InteractApp | taskStackPollResponse | Response to task stack poll | backend |
| InteractApp & Message | showAssistantreply | Get the agent's reply after processing the command, if any and render it.  | backend |
| InteractApp | setChatResponse | Get parsing status, logical form for this chat from semantic parser, history of past 5 commands | backend |
| VoxelWorld | getVoxelWorldInitialState | Get initial voxel world environment | frontend |
| VoxelWorld | setVoxelWorldInitialState | Set initial voxel world environment | backend |
| VoxelWorld | updateVoxelWorldState | Push updates to voxel world | backend |
| Settings & Message | connect | Display backend connection status | backend |
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
