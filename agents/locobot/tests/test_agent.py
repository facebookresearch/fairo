import socketio
import time
import os

# standard Python
sio = socketio.Client()

sio.connect("http://localhost:8000")

print("Connected to agent")

commands = [
    "move forward",
    "turn right",
]

for command in commands:
    print("sending command to agent: ", command)
    sio.emit('sendCommandToAgent', command)

time.sleep(5)
# wait for commands to be processed and maybe emit an error

try:
    print("Shutting down the agent")
    sio.emit("shutdown", {})

    print("disconnecting from the agent")
    sio.disconnect()
    print("disconnected")
except:
    os._exit(0)
