"""tests for environment interface."""

import pytest

from track import Action, Environment, State


def test_action_and_state_are_dataclasses():
    """action and state are frozen dataclasses usable as base types."""
    a = Action()
    s = State()
    assert a is not None
    assert s is not None


def test_environment_is_abstract():
    """environment cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Environment()  # pyright: ignore[reportAbstractUsage]
