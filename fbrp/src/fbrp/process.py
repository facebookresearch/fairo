import dataclasses
import inspect
import os
import pathlib
import typing

from fbrp import registrar


@dataclasses.dataclass
class ProcDef:
    name: str
    root: pathlib.Path
    rule_file: pathlib.Path
    runtime: "BaseRuntime"
    cfg: dict
    deps: typing.List[str]
    env: dict


def process(
    name: str,
    root: str = None,
    runtime: "BaseRuntime" = None,
    cfg: dict = {},
    deps: typing.List[str] = [],
    env: dict = {},
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
        env=env,
    )
