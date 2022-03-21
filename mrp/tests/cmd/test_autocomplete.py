from click import shell_completion
from mrp.cmd import _autocomplete
from unittest.mock import patch
import click
from mrp import life_cycle
from mrp import process_def


def make_autocomplete(suggestions):
    def impl(ctx, param, incomplete):
        return suggestions

    return impl


def get_completions(cmd, args, incomplete):
    complete = shell_completion.ShellComplete(cmd, {}, "", "")
    return [c.value for c in complete.get_completions(args, incomplete)]


def test_autocomplete_union():
    @click.command()
    @click.argument(
        "foo",
        shell_complete=_autocomplete.union(
            make_autocomplete(["aaa", "bbb", "ccc"]),
            make_autocomplete(["bbb", "ccc", "ddd"]),
        ),
    )
    def fake_cmd(foo):
        pass

    assert get_completions(fake_cmd, [], "") == ["aaa", "bbb", "ccc", "ddd"]


def test_autocomplete_intersection():
    @click.command()
    @click.argument(
        "foo",
        shell_complete=_autocomplete.intersection(
            make_autocomplete(["aaa", "bbb", "ccc"]),
            make_autocomplete(["bbb", "ccc", "ddd"]),
        ),
    )
    def fake_cmd(foo):
        pass

    assert get_completions(fake_cmd, [], "") == ["bbb", "ccc"]


def test_autocomplete_conditional():
    @click.command()
    @click.argument(
        "foo",
        shell_complete=_autocomplete.conditional(
            lambda ctx, unused_param, unused_incomplete: ctx.params["old"],
            make_autocomplete(["aa", "ab", "bb"]),
            make_autocomplete(["bb", "bc", "cc"]),
        ),
    )
    @click.option("-o", "--old", is_flag=True, default=False)
    def fake_cmd(foo):
        pass

    assert get_completions(fake_cmd, [], "") == ["bb", "bc", "cc"]
    assert get_completions(fake_cmd, ["-o"], "") == ["aa", "ab", "bb"]
    assert get_completions(fake_cmd, ["--old"], "") == ["aa", "ab", "bb"]


def test_autocomplete_defined_processes():
    with patch.dict(
        process_def.defined_processes,
        {"aa": None, "ab": None, "bb": None},
    ):

        @click.command()
        @click.argument(
            "foo",
            shell_complete=_autocomplete.defined_processes,
        )
        def fake_cmd(foo):
            pass

        assert get_completions(fake_cmd, [], "") == ["aa", "ab", "bb"]
        assert get_completions(fake_cmd, [], "a") == ["aa", "ab"]


def test_autocomplete_running_processes():
    class mock_system_state:
        def asdict(self):
            return {
                "procs": {
                    "a": {"state": "STARTED"},
                    "b": {"state": "STARTED"},
                    "c": {"state": "STOPPED"},
                }
            }

    with patch.object(life_cycle, "system_state", return_value=mock_system_state()):

        @click.command()
        @click.argument(
            "foo",
            shell_complete=_autocomplete.running_processes,
        )
        def fake_cmd(foo):
            pass

        assert get_completions(fake_cmd, [], "") == ["a", "b"]
