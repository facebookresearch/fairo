A high level design diagram of the current architecture of the dashboard is :

![dashboard architecture diagram](https://craftassist.s3-us-west-2.amazonaws.com/pubr/backend+architecture.png)

In the diagram above:
- frontend: The React based frontend that hosts other React subcomponents.
- StateManager: This module manages messaging between frontend and backend. We set the state of the frontend components through the data received socket.io connections between stateManager and backend. 
- backend : This is the agent process that sends and receives data from frontend. The agent process and its modules (perception, memory, mover, controller etc) can communicate using the the websocket transport.


Here's a flowchart of how the current dashboard works:
![dashboard flowchart](https://craftassist.s3-us-west-2.amazonaws.com/pubr/dashboard_flowchart.png)

In the figure above:
- The text on the communication arrows shows the names of the socket events being used in that communication.
- The names in rectangles are names of the frontend components.
- `Index.js` is the main React component that hosts other React Components in the following layout :
  - Left:
    - MainPane.js:
      - Top:
        - InteractApp: Lets you send a command to the assistant using voice or text. The commands once sent, are clickable to annotate if there was a language understanding error. This component also shows the status of the command execution by displaying agent reply or agent error messages.
      - Bottom:
        - Top left: LiveImage - Shows the rgb image stream
        - Top right: LiveImage - Shows the depth map
        - Bottom left: LiveObjects - Shows detector output
        - Bottom right: LiveHumans - Shows human pose estimation
  - Right:
    - Top right:
      - Memory2D: Visualizes the memory map on a 2D grid.
      - History: Shows the past 5 chat interactions through the dashboard.
      - QuerySemanticParser: Lets you query the semantic parser with some text.
      - TeachApp: Lets you do construct new programs for the assistant by assembling together blockly like blocks.
    - Bottom right:
      - Settings: Shows current settings, connection status, image quality, resolution etc
      - Navigator: Lets you navigate the assistant through keypress (up, down, left, right)
      - ObjectFixUp: Lets you annotate objects in dashboard.



Here's a snapshot of how this translates into the frontend:
![dashboard screenshot](https://craftassist.s3-us-west-2.amazonaws.com/pubr/dashboard_screenshot.png)