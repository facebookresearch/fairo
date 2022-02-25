from fbrp import process_def
import pytest


def test_invalid_env(capsys):
    with pytest.raises(SystemExit) as exit_info:
        process_def.process("foo", env={1: 2})
    assert exit_info.type == SystemExit
    assert exit_info.value.code == 1

    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert stderr == "fbrp process [foo] invalid. env is not dict[str, str]"
