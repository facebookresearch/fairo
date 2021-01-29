import os
import threading
from flask import Flask
import socketio
from flask_cors import cross_origin, CORS
import dlevent
import logging
import json
import random
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
    context = _get_context()
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
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        log.disabled = True

        log = logging.getLogger('socketio.server')
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

    app.run(ip, threaded=True, port=port, debug=False)


def start(web_root="web", ip="0.0.0.0", port=8000, quiet=True):
    t = threading.Thread(target=_dashboard_thread, args=(web_root, ip, port, quiet))
    t.start()
    context = _get_context()
    if context == _CONTEXT_COLAB or context == _CONTEXT_IPYTHON:
        # avoid race conditions when we do "run_all" on cells", give the
        # webserver thread to start and initialize sockets so that
        # `dlevent.sio` registers properly against those sockets if imported
        # immediately after the start() call
        import time
        time.sleep(5)

# Return values for `_get_context` (see that function's docs for
# details).
_CONTEXT_COLAB = "_CONTEXT_COLAB"
_CONTEXT_IPYTHON = "_CONTEXT_IPYTHON"
_CONTEXT_NONE = "_CONTEXT_NONE"

# this function is generously pulled from https://github.com/tensorflow/tensorboard/blob/d7fc1eab51ea6d0fb60e22eac7b271b4abc076e8/tensorboard/notebook.py#L53
# it is licensed under Apache 2.0 and Copyright 2017 The TensorFlow Authors.  All rights reserved.
def _get_context():
    """Determine the most specific context that we're in.

    Returns:
      _CONTEXT_COLAB: If in Colab with an IPython notebook context.
      _CONTEXT_IPYTHON: If not in Colab, but we are in an IPython notebook
        context (e.g., from running `jupyter notebook` at the command
        line).
      _CONTEXT_NONE: Otherwise (e.g., by running a Python script at the
        command-line or using the `ipython` interactive shell).
    """
    # In Colab, the `google.colab` module is available, but the shell
    # returned by `IPython.get_ipython` does not have a `get_trait`
    # method.
    try:
        import google.colab  # noqa: F401
        import IPython
    except ImportError:
        pass
    else:
        if IPython.get_ipython() is not None:
            # We'll assume that we're in a Colab notebook context.
            return _CONTEXT_COLAB

    # In an IPython command line shell or Jupyter notebook, we can
    # directly query whether we're in a notebook context.
    try:
        import IPython
    except ImportError:
        pass
    else:
        ipython = IPython.get_ipython()
        if ipython is not None and ipython.has_trait("kernel"):
            return _CONTEXT_IPYTHON

    # Otherwise, we're not in a known notebook context.
    return _CONTEXT_NONE


def display(port=8000, height=None, print_message=False, display_handle=None):
    if height is None:
        height = 800

    fn = {
        _CONTEXT_COLAB: _display_colab,
        _CONTEXT_IPYTHON: _display_ipython,
        # _CONTEXT_NONE: _display_cli,
    }[_get_context()]
    return fn(port=port, height=height, display_handle=display_handle)


def _display_ipython(port, height, display_handle):
    import IPython.display

    frame_id = "dldashboard-frame-{:08x}".format(random.getrandbits(64))
    shell = """
      <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">
      </iframe>
      <script>
        (function() {
          const frame = document.getElementById(%JSON_ID%);
          const url = new URL(%URL%, window.location);
          const port = %PORT%;
          if (port) {
            url.port = port;
          }
          frame.src = url;
        })();
      </script>
    """
    replacements = [
        ("%HTML_ID%", html_escape(frame_id, quote=True)),
        ("%JSON_ID%", json.dumps(frame_id)),
        ("%HEIGHT%", "%d" % height),
        ("%PORT%", "%d" % port),
        ("%URL%", json.dumps("/")),
    ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)
    iframe = IPython.display.HTML(shell)
    if display_handle:
        display_handle.update(iframe)
    else:
        IPython.display.display(iframe)


def _display_colab(port, height, display_handle):
    """Display a TensorBoard instance in a Colab output frame.

    The Colab VM is not directly exposed to the network, so the Colab
    runtime provides a service worker tunnel to proxy requests from the
    end user's browser through to servers running on the Colab VM: the
    output frame may issue requests to https://localhost:<port> (HTTPS
    only), which will be forwarded to the specified port on the VM.

    It does not suffice to create an `iframe` and let the service worker
    redirect its traffic (`<iframe src="https://localhost:6006">`),
    because for security reasons service workers cannot intercept iframe
    traffic. Instead, we manually fetch the TensorBoard index page with an
    XHR in the output frame, and inject the raw HTML into `document.body`.

    By default, the TensorBoard web app requests resources against
    relative paths, like `./data/logdir`. Within the output frame, these
    requests must instead hit `https://localhost:<port>/data/logdir`. To
    redirect them, we change the document base URI, which transparently
    affects all requests (XHRs and resources alike).
    """
    import IPython.display

    shell = """
        (async () => {
            const url = new URL(await google.colab.kernel.proxyPort(%PORT%, {'cache': true}));
            const iframe = document.createElement('iframe');
            iframe.src = url;
            iframe.setAttribute('width', '100%');
            iframe.setAttribute('height', '%HEIGHT%');
            iframe.setAttribute('frameborder', 0);
            document.body.appendChild(iframe);
        })();
    """
    replacements = [
        ("%PORT%", "%d" % port),
        ("%HEIGHT%", "%d" % height),
    ]
    for (k, v) in replacements:
        shell = shell.replace(k, v)
    script = IPython.display.Javascript(shell)

    if display_handle:
        display_handle.update(script)
    else:
        IPython.display.display(script)

