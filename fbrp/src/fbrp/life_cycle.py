import threading
from fbrp import util
import a0
import dataclasses
import enum
import json
import typing


def from_dict(cls, obj):
    if isinstance(obj, (tuple, list)):
        return [from_dict(cls.__args__[0], f) for f in obj]
    return cls(**{k: from_dict(cls.__annotations__[k], v) for k, v in obj.items()})


@dataclasses.dataclass
class ProcInfo:
    class Ask(enum.Enum):
        UP = "UP"
        DOWN = "DOWN"

    class State(enum.Enum):
        STARTING = "STARTING"
        STARTED = "STARTED"
        STOPPING = "STOPPING"
        STOPPED = "STOPPED"

    ask: Ask
    state: State
    return_code: int
    launcher_running: bool


@dataclasses.dataclass
class SystemState:
    procs: typing.Mapping[str, ProcInfo]


def system_state() -> SystemState:
    return from_dict(SystemState, json.loads(a0.Cfg("fbrp/state").read().payload))


def system_state_watcher(callback) -> a0.CfgWatcher:
    def callback_wrapper(pkt):
        callback(from_dict(SystemState, json.loads(pkt.payload)))

    return a0.CfgWatcher("fbrp/state", callback_wrapper)


def ask(proc_name, ask_):
    a0.Cfg("fbrp/state").mergepatch({proc_name: {"ask": ask_}})
    with util.common_env_context(proc_name):
        a0.Publisher(f"fbrp/control/{proc_name}").pub(json.dumps({"ask": ask_}))
