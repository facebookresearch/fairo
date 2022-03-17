from fbrp import process_def
import pytest


def test_invalid_env():
    with pytest.raises(ValueError) as err:
        process_def.process("foo", env={1: 2})
    assert err.type == ValueError
    assert str(err.value) == "fbrp process [foo] invalid. env is not dict[str, str]"
