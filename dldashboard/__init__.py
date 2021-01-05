import os
import threading
from flask import Flask
import socketio
from flask_cors import cross_origin, CORS
import dlevent

app = None


def _dashboard_thread(web_root, ip, port):
    global app
    root_dir = os.path.abspath(os.path.dirname(__file__))
    static_folder = os.path.join(root_dir, web_root, "build")
    print("static_folder:", static_folder)
    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
    dlevent.sio = sio

    CORS(app, resources={r"*": {"origins": "*"}})

    @app.route("/")
    @cross_origin(origin="*")
    def index():
        return app.send_static_file("index.html")

    if os.getenv("MCDASHBOARD_PORT"):
        port = os.getenv("MCDASHBOARD_PORT")
        print("setting MC dashboard port from env variable MCDASHBOARD_PORT={}".format(port))

    if os.getenv("MCDASHBOARD_IP"):
        ip = os.getenv("MCDASHBOARD_IP")
        print("setting MC dashboard ip from env variable MCDASHBOARD_IP={}".format(ip))

    app.run(ip, threaded=True, port=port, debug=False)


def start(web_root="web", ip="0.0.0.0", port=8000):
    t = threading.Thread(target=_dashboard_thread, args=(web_root, ip, port))
    t.start()
