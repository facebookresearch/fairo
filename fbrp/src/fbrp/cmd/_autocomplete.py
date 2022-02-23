from fbrp import life_cycle
from fbrp import process_def
import a0
import glob
import os


def union(*fns):
    def impl(ctx, param, incomplete):
        builder = set()
        for fn in fns:
            builder |= set(fn(ctx, param, incomplete))
        return sorted(builder)

    return impl


def intersection(*fns):
    def impl(ctx, param, incomplete):
        builder = None
        for fn in fns:
            if builder is None:
                builder = set(fn(ctx, param, incomplete))
            else:
                builder &= set(fn(ctx, param, incomplete))
        return sorted(builder or [])

    return impl


def conditional(predicate, ontrue, onfalse):
    def impl(ctx, param, incomplete):
        if predicate(ctx, param, incomplete):
            return ontrue(ctx, param, incomplete)
        else:
            return onfalse(ctx, param, incomplete)

    return impl


def defined_processes(ctx, param, incomplete):
    return [
        name
        for name, proc_def in process_def.defined_processes.items()
        if name.startswith(incomplete)
    ]


def running_processes(ctx, param, incomplete):
    state = life_cycle.system_state().asdict()
    return [
        name
        for name, info in state.get("procs", {}).items()
        if info.get("state") == "STARTED" and name.startswith(incomplete)
    ]


def alephzero_topics(protocol):
    def fn(ctx, param, incomplete):
        topics = []
        # TODO(lshamis): Use the a0 pathglob.
        detected = glob.glob(
            os.path.join(a0.env.root(), f"**/*.{protocol}.a0"), recursive=True
        )
        for abspath in detected:
            relpath = os.path.relpath(abspath, a0.env.root())
            topic = relpath[: -len(f".{protocol}.a0")]
            topics.append(topic)

        return [name for name in topics if name.startswith(incomplete)]

    return fn
