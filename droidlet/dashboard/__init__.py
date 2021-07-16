import os
import threading
from droidlet.parallel import PropagatingThread
from flask import Flask
import socketio
from flask_cors import cross_origin, CORS
from droidlet import event
import logging
import json
import random
import ssl

# https://github.com/miguelgrinberg/python-engineio/issues/142
from engineio.payload import Payload
Payload.max_decode_packets = 5000000

try:
    import html

    html_escape = html.escape
    del html
except ImportError:
    import cgi

    html_escape = cgi.escape
    del cgi

_dashboard_app = None

def _dashboard_thread(web_root, ip, port, socketio_initialized, quiet=True):
    global _dashboard_app
    root_dir = os.path.abspath(os.path.dirname(__file__))
    static_folder = os.path.join(root_dir, web_root, "build")
    if not quiet:
        print("static_folder:", static_folder)

    _dashboard_app = Flask(__name__, static_folder=static_folder, static_url_path="")
    sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
    _dashboard_app.wsgi_app = socketio.WSGIApp(sio, _dashboard_app.wsgi_app)
    event.sio = sio
    socketio_initialized.set()

    CORS(_dashboard_app, resources={r"*": {"origins": "*"}})

    if quiet:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        log = logging.getLogger("socketio.server")
        log.setLevel(logging.ERROR)
        log.disabled = True
        _dashboard_app.logger.disabled = True

    @_dashboard_app.route("/")
    @cross_origin(origin="*")
    def index():
        return _dashboard_app.send_static_file("index.html")

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

    _dashboard_app.run(ip, threaded=True, port=port, ssl_context=ssl_context, debug=False)


def start(web_root="web", ip="0.0.0.0", port=8000, quiet=True):
    socketio_initialized = threading.Event()
    _dashboard_app_thread = PropagatingThread(target=_dashboard_thread, args=(web_root, ip, port, socketio_initialized, quiet), daemon=True)
    _dashboard_app_thread.start()

    # avoid race conditions, wait for the thread to start and set the socketio objec
    socketio_initialized.wait()
