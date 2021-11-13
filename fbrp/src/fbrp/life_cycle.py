import threading
from fbrp import util
import a0
import dataclasses
import enum
import json
import typing
import types

_TOPIC = "fbrp/state"
_CFG = a0.Cfg(_TOPIC)


class Ask(enum.Enum):
    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"


class State(enum.Enum):
    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    STARTED = "STARTED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


@dataclasses.dataclass(frozen=True)
class ProcInfo:
    ask: Ask
    state: State
    return_code: int
    launcher_running: bool

    def asdict(self):
        return dict(
            ask=self.ask.value,
            state=self.state.value,
            return_code=self.return_code,
            launcher_running=self.launcher_running,
        )

    @classmethod
    def fromdict(cls, dict_):
        return cls(
            ask=Ask(dict_.get("ask", Ask.NONE)),
            state=State(dict_.get("state", State.UNKNOWN)),
            return_code=dict_.get("return_code", 0),
            launcher_running=dict_.get("launcher_running", False),
        )


@dataclasses.dataclass
class SystemState:
    procs: typing.Mapping[str, ProcInfo]

    def asdict(self):
        return {"procs": {k: v.asdict() for k, v in self.procs.items()}}

    @classmethod
    def fromdict(cls, dict_):
        return SystemState(
            procs={k: ProcInfo.fromdict(v) for k, v in dict_.get("procs", {}).items()}
        )


def _ensure_setup() -> None:
    _CFG.write_if_empty(json.dumps(SystemState(procs={}).asdict()))


def system_state() -> SystemState:
    _ensure_setup()
    return SystemState.fromdict(json.loads(_CFG.read().payload))


def proc_info(proc_name) -> ProcInfo:
    return system_state().procs[proc_name]


def system_state_watcher(callback) -> a0.CfgWatcher:
    _ensure_setup()

    def callback_wrapper(pkt):
        callback(SystemState.fromdict(json.loads(pkt.payload)))

    return a0.CfgWatcher(_TOPIC, callback_wrapper)


async def aio_system_state_watcher():
    _ensure_setup()
    async for pkt in a0.aio_cfg(_TOPIC):
        yield SystemState.fromdict(json.loads(pkt.payload))


def proc_info_watcher(proc_name, callback) -> a0.CfgWatcher:
    ns = types.SimpleNamespace()
    ns.last_proc_info = None

    def callback_wrapper(system_state):
        if ns.last_proc_info != system_state.procs.get(proc_name):
            ns.last_proc_info = system_state.procs[proc_name]
            callback(system_state.procs[proc_name])

    return system_state_watcher(callback_wrapper)


async def aio_proc_info_watcher(proc_name):
    ns = types.SimpleNamespace()
    ns.last_proc_info = None

    async for system_state in aio_system_state_watcher():
        if ns.last_proc_info != system_state.procs.get(proc_name):
            ns.last_proc_info = system_state.procs[proc_name]
            yield system_state.procs[proc_name]


def set_ask(proc_name, ask):
    _CFG.mergepatch({"procs": {proc_name: {"ask": ask.value}}})


def set_state(proc_name, state, return_code=0):
    _CFG.mergepatch(
        {"procs": {proc_name: {"state": state.value, "return_code": return_code}}}
    )


def set_launcher_running(proc_name, launcher_running):
    _CFG.mergepatch({"procs": {proc_name: {"launcher_running": launcher_running}}})
