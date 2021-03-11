#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from flask import (
    Flask,
    Blueprint,
    render_template,
    abort,
    request,
    send_from_directory,
    jsonify,
)
from geventwebsocket import (
    WebSocketServer,
    WebSocketApplication,
    Resource,
    WebSocketError,
)
from uuid import uuid4
import time
import json
import os
from werkzeug.utils import secure_filename

from threading import Event

from typing import Dict, Tuple, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from geventwebsocket.handler import Client
    from geventwebsocket.websocket import WebSocket

# Constants

PACKET_TYPE_INIT_DATA = "initial_data_send"
PACKET_TYPE_AGENT_ACTION = "agent_action"
PACKET_TYPE_REQUEST_ACTION = "request_act"
PACKET_TYPE_UPDATE_AGENT_STATUS = "update_status"
PACKET_TYPE_NEW_AGENT = "register_agent"
PACKET_TYPE_NEW_WORKER = "register_worker"
PACKET_TYPE_REQUEST_AGENT_STATUS = "request_status"
PACKET_TYPE_RETURN_AGENT_STATUS = "return_status"
PACKET_TYPE_GET_INIT_DATA = "init_data_request"
PACKET_TYPE_ALIVE = "alive"
PACKET_TYPE_PROVIDER_DETAILS = "provider_details"
PACKET_TYPE_SUBMIT_ONBOARDING = "submit_onboarding"
PACKET_TYPE_ERROR_LOG = "log_error"


SYSTEM_CHANNEL_ID = "mephisto"  # TODO pull from somewhere
SERVER_CHANNEL_ID = "mephisto_server"

FAILED_RECONNECT_TIME = 10  # seconds
UPLOAD_FOLDER = "/tmp/"
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}

STATUS_INIT = "none"
STATUS_CONNECTED = "connected"
STATUS_DISCONNECTED = "disconnect"
STATUS_COMPLETED = "completed"

PACKET_TYPE_HEARTBEAT = "heartbeat"

DEBUG = False


# Main application setup
mephisto_router = Blueprint(
    "mephisto",
    __name__,
    static_folder="static",
)


def debug_log(*args):
    """
    Log only if debugging is enabled

    Explicitly does not use the regular Mephisto logging framework as we
    may want to deploy this on a server that doesn't have Mephisto installed,
    and we can keep package size low this way.
    """
    if DEBUG:
        print(*args)


def js_time(python_time: float) -> int:
    """Convert python time to js time, as the mephisto-task package expects"""
    return int(python_time * 1000)


# Socket and agent details
class LocalAgentState:
    """
    Keeps track of a connected agent over their lifecycle interacting with the router
    """

    def __init__(self, agent_id: str):
        """Initialize an object to track the lifecycle of a connection"""
        self.status = STATUS_INIT
        self.agent_id = agent_id
        self.state: Dict[str, Any] = {"wants_act": False, "done_text": None}
        self.is_alive = False
        self.disconnect_time = 0

    def __str__(self):
        return f"Agent({self.agent_id}): {self.status}, {self.state}"


class MephistoRouterState:
    def __init__(self):
        self.agent_id_to_client: Dict[str, "Client"] = {}
        self.client_id_to_agent: Dict[str, LocalAgentState] = {}
        self.mephisto_socket: Optional["WebSocket"] = None
        self.agent_id_to_agent: Dict[str, LocalAgentState] = {}
        self.pending_provider_requests: Dict[str, bool] = {}
        self.received_provider_responses: Dict[str, Dict[str, Any]] = {}
        self.last_mephisto_ping: float = time.time()


mephisto_router_app: Optional["MephistoRouter"] = None
mephisto_router_state: Optional["MephistoRouterState"] = None


def register_router_application(router: "MephistoRouter") -> "MephistoRouterState":
    """
    Register a routing application with the global state,
    such that HTTP requests can access it and so that
    all websocket routers share the same state.

    Returns the global router state
    """
    global mephisto_router_app, mephisto_router_state
    mephisto_router_app = router
    if mephisto_router_state is None:
        mephisto_router_state = MephistoRouterState()
    return mephisto_router_state


class MephistoRouter(WebSocketApplication):
    """
    Base implementation of a websocket server that handles
    all of the socket based IO for mephisto-task
    """

    def __init__(self, *args, **kwargs):
        """Initialize with the gloabl state of MephistoRouters"""
        super().__init__(*args, **kwargs)
        self.mephisto_state = register_router_application(self)

    def _send_message(self, socket: "WebSocket", packet: Dict[str, Any]) -> None:
        """Send the given message through the given socket"""
        if not socket:
            # We should be passing a socket, even if it's closed...
            debug_log("No socket to send packet to", packet)
            return

        if socket.closed:
            # Socket is already closed, noop
            return

        socket.send(json.dumps(packet))

    def _find_or_create_agent(self, agent_id: str) -> "LocalAgentState":
        """Get or create an agent state for the given id"""
        state = self.mephisto_state
        agent = state.agent_id_to_agent.get(agent_id)
        if agent is None:
            agent = LocalAgentState(agent_id)
            state.agent_id_to_agent[agent_id] = agent
        return agent

    def _handle_alive(self, client: "Client", alive_packet: Dict[str, Any]) -> None:
        """
        On alive, find out who the sender is, and register
        them as correctly here.
        """
        state = self.mephisto_state
        if alive_packet["sender_id"] == SYSTEM_CHANNEL_ID:
            state.mephisto_socket = client.ws
        else:
            agent_id = alive_packet["sender_id"]
            agent = self._find_or_create_agent(agent_id)
            agent.is_alive = True
            state.agent_id_to_client[agent_id] = client
            state.client_id_to_agent[client.mephisto_id] = agent

    def _handle_get_agent_status(self, agent_status_packet: Dict[str, Any]) -> None:
        """
        On a get agent status request, forward the request to all tracked agents
        then without waiting for the response, respond to the core mephisto server
        with the current status of each.

        May return semi-stale information, but is non-blocking
        """
        state = self.mephisto_state
        state.last_mephisto_ping = time.time()
        agent_statuses = {}
        for agent_id in state.agent_id_to_agent.keys():
            agent = self._find_or_create_agent(agent_id)
            if not agent.is_alive and agent.status != STATUS_DISCONNECTED:
                self._followup_possible_disconnect(agent)
            agent_statuses[agent_id] = state.agent_id_to_agent[agent_id].status
            ping_packet = {
                "packet_type": PACKET_TYPE_REQUEST_AGENT_STATUS,
                "sender_id": SYSTEM_CHANNEL_ID,
                "receiver_id": agent_id,
                "data": None,
            }
            self._handle_forward(ping_packet)
        packet = {
            "packet_type": PACKET_TYPE_RETURN_AGENT_STATUS,
            "sender_id": SERVER_CHANNEL_ID,
            "receiver_id": SYSTEM_CHANNEL_ID,
            "data": agent_statuses,
        }
        self._handle_forward(packet)

    def _get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Return the agent state for a given tracked agent"""
        agent = self._find_or_create_agent(agent_id)
        return agent.state

    def _handle_update_local_status(self, status_packet: Dict[str, Any]) -> None:
        """Update the local agent status given a status packet"""
        agent_id = status_packet["receiver_id"]
        agent = self._find_or_create_agent(agent_id)
        if status_packet["data"].get("status") is not None:
            agent.status = status_packet["data"]["status"]
        if status_packet["data"].get("state") is not None:
            agent.state.update(status_packet["data"]["state"])

    def _update_wanted_acts(self, agent_id: str, wants_act: bool) -> None:
        """Update the wanted acts flag for a given agent"""
        agent = self._find_or_create_agent(agent_id)
        agent.state["wants_act"] = wants_act

    def _handle_forward(self, packet: Dict[str, Any]) -> None:
        """Handle forwarding the given packet to the included receiver_id"""
        if packet["receiver_id"] == SYSTEM_CHANNEL_ID:
            debug_log("Adding message to mephisto queue", packet)
            socket = self.mephisto_state.mephisto_socket
        else:
            agent_id = packet["receiver_id"]
            client = self.mephisto_state.agent_id_to_client.get(agent_id)
            if client is None:
                debug_log(f"No agent found to send {packet} to")
                return
            socket = client.ws
        self._send_message(socket, packet)

    def _followup_possible_disconnect(self, agent: LocalAgentState) -> None:
        """Check to see if the given agent is disconnected"""
        if agent.disconnect_time == 0:
            return  # Agent never disconnected, isn't live
        if time.time() - agent.disconnect_time > FAILED_RECONNECT_TIME:
            agent.status = STATUS_DISCONNECTED
            debug_log("Agent disconnected", agent)

    def _send_status_for_agent(self, agent_id: str) -> None:
        """
        Send a packet that updates the client status for the given agent,
        pushing them the most recent local status.
        """
        agent = self._find_or_create_agent(agent_id)
        packet = {
            "packet_type": PACKET_TYPE_UPDATE_AGENT_STATUS,
            "sender_id": SERVER_CHANNEL_ID,
            "receiver_id": agent_id,
            "data": {
                "status": agent.status,
                "state": agent.state,
            },
        }
        self._handle_forward(packet)

    def on_open(self) -> None:
        """
        Initialize a new client connection, and give them a uuid to refer to
        """
        current_client = self.ws.handler.active_client
        debug_log("Some client connected!", current_client)
        current_client.mephisto_id = str(uuid4())

    def on_message(self, message: str) -> None:
        """
        Determine the type of message, and then handle via the correct handler
        """
        if message is None:
            return

        state = self.mephisto_state
        current_client = self.ws.handler.active_client
        client = current_client
        packet = json.loads(message)
        if packet["packet_type"] == PACKET_TYPE_REQUEST_AGENT_STATUS:
            debug_log("Mephisto requesting status")
            self._handle_get_agent_status(packet)
        elif packet["packet_type"] == PACKET_TYPE_AGENT_ACTION:
            debug_log("Agent action: ", packet)
            self._handle_forward(packet)
            if packet["receiver_id"] == SYSTEM_CHANNEL_ID:
                self._update_wanted_acts(packet["sender_id"], False)
                self._send_status_for_agent(packet["sender_id"])
        elif packet["packet_type"] == PACKET_TYPE_ERROR_LOG:
            self._handle_forward(packet)
        elif packet["packet_type"] == PACKET_TYPE_ALIVE:
            debug_log("Agent alive: ", packet)
            self._handle_alive(self.ws.handler.active_client, packet)
        elif packet["packet_type"] == PACKET_TYPE_UPDATE_AGENT_STATUS:
            debug_log("Update agent status", packet)
            self._handle_update_local_status(packet)
            packet["data"]["state"] = self._get_agent_state(packet["receiver_id"])
            self._handle_forward(packet)
        elif packet["packet_type"] == PACKET_TYPE_REQUEST_ACTION:
            debug_log("Requesting act", packet)
            agent_id = packet["receiver_id"]
            self._update_wanted_acts(agent_id, True)
            self._send_status_for_agent(agent_id)
        elif packet["packet_type"] in [
            PACKET_TYPE_PROVIDER_DETAILS,
            PACKET_TYPE_INIT_DATA,
        ]:
            request_id = packet["data"].get("request_id")
            if request_id is None:
                request_id = packet["receiver_id"]
            res_event = state.pending_provider_requests.get(request_id)
            if res_event is not None:
                state.received_provider_responses[request_id] = packet
                del state.pending_provider_requests[request_id]
        elif packet["packet_type"] == PACKET_TYPE_HEARTBEAT:
            packet["data"] = {"last_mephisto_ping": js_time(state.last_mephisto_ping)}
            agent_id = packet["sender_id"]
            packet["sender_id"] = packet["receiver_id"]
            packet["receiver_id"] = agent_id
            agent = state.agent_id_to_agent.get(agent_id)
            if agent is not None:
                agent.is_alive = True
                packet["data"]["status"] = agent.status
                packet["data"]["state"] = agent.state
                if state.agent_id_to_client.get(agent.agent_id) != client:
                    # Not communicating to the correct socket, update
                    debug_log("Updating client for ", agent)
                    state.agent_id_to_client[agent.agent_id] = client
                    state.client_id_to_agent[client.mephisto_id] = agent
            self._handle_forward(packet)
        else:
            debug_log("Unknown message", packet)

    def on_close(self, reason: Any) -> None:
        """Mark a socket dead for a LocalAgentState, give time to reconnect"""
        client = self.ws.handler.active_client
        debug_log("Some client disconnected!", client.mephisto_id)
        agent = self.mephisto_state.client_id_to_agent.get(client.mephisto_id)
        if agent is None:
            return  # Agent not being tracked
        agent.is_alive = False
        agent.disconnect_time = time.time()

    def get_default_provider_request_packet(
        self, request_type: str, provider_data: Dict[str, Any]
    ):
        """Create a packet used for simple provider requests"""
        request_id = str(uuid4())
        return {
            "packet_type": request_type,
            "sender_id": SERVER_CHANNEL_ID,
            "receiver_id": SYSTEM_CHANNEL_ID,
            "data": {
                "provider_data": provider_data,
                "request_id": request_id,
            },
        }

    def make_provider_request(
        self, request_packet: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make a request to the core Mephisto server, and then await the response"""
        request_id = request_packet["data"]["request_id"]

        self.mephisto_state.pending_provider_requests[request_id] = True
        self._send_message(self.mephisto_state.mephisto_socket, request_packet)
        start_time = time.time()
        res = None
        while time.time() - start_time < 30 and res is None:
            res = self.mephisto_state.received_provider_responses.get(request_id)
            time.sleep(0.01)
        if res is not None:
            del self.mephisto_state.received_provider_responses[request_id]
        return res


def handle_provider_request(request_type: str, data: Dict[str, Any]):
    """Wrapper for provider requests that handles extracting the result and timing out"""
    provider_data = data["provider_data"]
    packet = mephisto_router_app.get_default_provider_request_packet(
        request_type, provider_data
    )
    res = mephisto_router_app.make_provider_request(packet)
    if res is not None:
        return jsonify(res)
    else:
        # Timed out waiting for Mephisto to respond
        abort(408)
        return None


@mephisto_router.route("/initial_task_data", methods=["POST"])
def initial_task_data():
    data = request.get_json()
    return handle_provider_request(PACKET_TYPE_GET_INIT_DATA, data)


@mephisto_router.route("/register_worker", methods=["POST"])
def register_worker():
    data = request.get_json()
    return handle_provider_request(PACKET_TYPE_NEW_WORKER, data)


@mephisto_router.route("/request_agent", methods=["POST"])
def request_agent():
    data = request.get_json()
    return handle_provider_request(PACKET_TYPE_NEW_AGENT, data)


@mephisto_router.route("/submit_onboarding", methods=["POST"])
def submit_onboarding():
    """
    Parse onboarding as if it were a request sent from the
    active agent, rather than coming as a request from the router.
    """
    data = request.get_json()
    provider_data = data["provider_data"]
    agent_id = provider_data["USED_AGENT_ID"]
    del provider_data["USED_AGENT_ID"]

    # Construct and send onboarding submission packet
    packet = mephisto_router_app.get_default_provider_request_packet(
        PACKET_TYPE_SUBMIT_ONBOARDING, provider_data
    )
    packet["sender_id"] = agent_id
    res = mephisto_router_app.make_provider_request(packet)
    if res is not None:
        return jsonify(res)
    else:
        # Timed out waiting for Mephisto to respond
        abort(408)
        return None


@mephisto_router.route("/submit_task", methods=["POST"])
def submit_task():
    """Parse task submission as if it were an act"""
    provider_data = request.get_json()
    filenames = []
    if provider_data is None:
        # Multipart form submit
        provider_data = request.form.to_dict()
        files = request.files.to_dict()
        if len(files) > 0:
            timestamp = int(time.time())
            rand = str(uuid4())[:8]
            for filename, filepoint in files.items():
                full_name = f"{timestamp}-{rand}-{secure_filename(filename)}"
                filepoint.save(os.path.join("/tmp/", full_name))
                filenames.append({"filename": full_name})

    agent_id = provider_data["USED_AGENT_ID"]
    del provider_data["USED_AGENT_ID"]
    packet = {
        "packet_type": PACKET_TYPE_AGENT_ACTION,
        "sender_id": agent_id,
        "receiver_id": SYSTEM_CHANNEL_ID,
        "data": {
            "task_data": provider_data,
            "MEPHISTO_is_submit": True,
            "files": filenames,
        },
    }
    mephisto_router_app._handle_forward(packet)
    return jsonify({"status": "Error log sent!"})


@mephisto_router.route("/log_error", methods=["POST"])
def log_error():
    data = request.get_json()
    provider_data = data["provider_data"]
    agent_id = provider_data["USED_AGENT_ID"]
    del provider_data["USED_AGENT_ID"]
    packet = {
        "packet_type": PACKET_TYPE_ERROR_LOG,
        "sender_id": agent_id,
        "receiver_id": SYSTEM_CHANNEL_ID,
        "data": provider_data,
    }
    mephisto_router_app._handle_forward(packet)
    return jsonify({"status": "Error log sent!"})


@mephisto_router.route("/is_alive", methods=["GET"])
def is_alive():
    return jsonify({"status": "Alive!"})


@mephisto_router.route("/get_timestamp", methods=["GET"])
def get_timestamp():
    return jsonify({"timestamp": time.time()})


@mephisto_router.route("/download_file/<filename>", methods=["GET"])
def download_file(filename):
    try:
        return send_from_directory("/tmp/", filename)
    except:
        abort(404)


@mephisto_router.route("/")
def show_index():
    try:
        return send_from_directory("static", "index.html")
    except:
        abort(404)


@mephisto_router.route("/<res>")
def get_static(res):
    try:
        return send_from_directory("static", res)
    except:
        abort(404)


@mephisto_router.after_request
def add_header(r):
    """
    Add headers to prevent caching, as this server may be used in local
    development or with the same address but different contents
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r
