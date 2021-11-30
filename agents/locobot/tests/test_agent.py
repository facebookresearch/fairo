import socketio
import time
import os

# standard Python
sio = socketio.Client()

connect_count = 0
connect_max_tries = 10
connect_delay = 5 # seconds
while True:
    try:
        sio.connect("http://localhost:8000")
        print("Connected to agent")
        break
    except socketio.exceptions.ConnectionError:
        connect_count = connect_count + 1
        if connect_count >= connect_max_tries:
            raise
        print("Failed connecting for the {} / {} time. "
              "Retrying in {} seconds".format(connect_count, connect_max_tries, connect_delay))
        sio.sleep(connect_delay)

agent_running = False

@sio.on("currentCount")
def currentCount(data):
    global agent_running
    if data > 1:
        agent_running = True

timeout = 60 # seconds
delay = 1 # second
# wait for agent loop to actually start executing
for i in range(timeout):
    print("Waiting for agent to step once {} / {}".format(i, timeout))
    sio.emit("get_count", i)
    sio.sleep(delay)
    if agent_running:
        break
if not agent_running:
    print("agent never started executing "
          "even after {} seconds".format(timeout))
    os._exit(1)

commands = [
    "move forward",
    "turn right",
]

for command in commands:
    print("sending command to agent: ", command)
    sio.emit('sendCommandToAgent', command)

agent_objects = False

@sio.on("objects")
def objects(data):
    global agent_objects
    if data["image"] != -1:
        agent_objects = True
        print("Got object detection output from agent")
    else:
        print("Waiting for object detection output", data)

for i in range(timeout):
    print("Waiting for objects from agent {} / {}".format(i, timeout))
    if agent_objects:
        break
    sio.sleep(delay)

if not agent_objects:
    print("agent not processing object detection "
          "even after {} seconds".format(timeout))
    os._exit(1)

try:
    print("Shutting down the agent")
    sio.emit("shutdown", {})

    print("disconnecting from the agent")
    sio.disconnect()
    print("disconnected")
except:
    os._exit(0)
