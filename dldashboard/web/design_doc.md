A high level design diagram of the current architecture of the dashboard is :
<p align="center">
    ![dashboard architecture diagram](https://s3.console.aws.amazon.com/s3/object/craftassist?region=us-west-2&prefix=pubr/backend+architecture.png)
</p>

Here:
- frontend: The React based frontend that hosts other React subcomponents.
- StateManager: This module manages messaging between frontend and backend. We set the state of the frontend components through the data received socket.io connections between stateManager and backend. 
- backend : This is the agent process that sends and receives data from frontend. The agent process and its modules (perception, memory, mover, controller etc) can communicate using the the websocket transport.


Here's a flowchart of how the current dashboard works:
<p align="center">
    ![dashboard flowchart](https://s3.console.aws.amazon.com/s3/object/craftassist?region=us-west-2&prefix=pubr/dashboard_flowchart.png)
</p>

The text on the communication arrows shows the names of the socket events being handled currently.
