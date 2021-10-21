import dataclasses
import inspect
import os
import pathlib
import typing

from facebook_robotics_platform.setup import registrar


@dataclasses.dataclass
class ProcDef:
    name: str
    root: pathlib.Path
    rule_file: pathlib.Path
    runtime: "BaseRuntime"
    cfg: dict
    deps: typing.List[str]


def process(
    name: str,
    root: str = None,
    runtime: "BaseRuntime" = None,
    cfg: dict = {},
    deps: typing.List[str] = [],
):
    if name in registrar.defined_processes:
        raise ValueError(f"fbrp.process({name=}) defined multiple times.")

    rule_file = os.path.abspath(inspect.stack()[1].filename)
    rule_dir = os.path.dirname(rule_file)

    if root:
        root = os.path.normpath(os.path.join(rule_dir, root))
    else:
        root = rule_dir

    registrar.defined_processes[name] = ProcDef(
        name=name,
        root=root,
        runtime=runtime,
        cfg=cfg,
        deps=deps,
        rule_file=rule_file,
    )
