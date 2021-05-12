import os
import threading
from flask import Flask
import socketio
from flask_cors import cross_origin, CORS
import dlevent
import logging
import json
import random
import ssl

try:
    import html

    html_escape = html.escape
    del html
except ImportError:
    import cgi

    html_escape = cgi.escape
    del cgi

app = None


def _dashboard_thread(web_root, ip, port, quiet=True):
    global app
    root_dir = os.path.abspath(os.path.dirname(__file__))
    static_folder = os.path.join(root_dir, web_root, "build")
    if not quiet:
        print("static_folder:", static_folder)

    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
    dlevent.sio = sio

    CORS(app, resources={r"*": {"origins": "*"}})

    if quiet:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        log = logging.getLogger("socketio.server")
        log.setLevel(logging.ERROR)
        log.disabled = True
        app.logger.disabled = True

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

    mcdashboard_ssl_cert = os.getenv("MCDASHBOARD_SSL_CERT")
    mcdashboard_ssl_pkey = os.getenv("MCDASHBOARD_SSL_PKEY")
    ssl_context = None
    if (
        mcdashboard_ssl_cert
        and mcdashboard_ssl_pkey
        and os.path.isfile(mcdashboard_ssl_cert)
        and os.path.isfile(mcdashboard_ssl_pkey)
    ):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.load_cert_chain(mcdashboard_ssl_cert, mcdashboard_ssl_pkey)
        print("SSL certificate found, enabling https")

    app.run(ip, threaded=True, port=port, ssl_context=ssl_context, debug=False)


def start(web_root="web", ip="0.0.0.0", port=8000, quiet=True):
    t = threading.Thread(target=_dashboard_thread, args=(web_root, ip, port, quiet))
    t.start()
    # avoid race conditions, wait for the thread to start and set the socketio object
    # TODO: rewrite this using thread signaling instead of a dumb sleep
    import time

    time.sleep(3)
