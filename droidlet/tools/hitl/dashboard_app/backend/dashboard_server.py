from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("my message")
def handle_message(data):
    print("received message: " + data)
    emit("my message", ["from server: copy that", "1", "2", "3"])


if __name__ == "__main__":
    socketio.run(app)
