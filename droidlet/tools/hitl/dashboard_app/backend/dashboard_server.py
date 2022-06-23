import logging
from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import get_job_list
from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("get_job_list")
def handle_message():
    print("Request received: get_job_list")
    job_list = get_job_list()
    print(job_list)
    emit("get_job_list", job_list)

if __name__ == "__main__":
    socketio.run(app)
