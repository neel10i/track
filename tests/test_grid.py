"""tests for grid environment."""

import pytest

from track import GridEnvironment, GridState, Move, PickupKey


def test_initial_state():
    """initial state has agent at start, key on key_pos."""
    env = GridEnvironment(
        width=3,
        height=3,
        obstacles=frozenset(),
        key_pos=(1, 1),
        goal_pos=(2, 2),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    assert s.agent_pos == (0, 0)
    assert s.key_pos == (1, 1)
    assert s.goal_pos == (2, 2)
    assert s.width == 3 and s.height == 3


def test_move_in_bounds():
    """move updates agent position."""
    env = GridEnvironment(
        width=3,
        height=3,
        obstacles=frozenset(),
        key_pos=(2, 2),
        goal_pos=(2, 2),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    s2, _ = env.step(s, Move(1, 0))
    assert s2.agent_pos == (1, 0)
    s3, _ = env.step(s2, Move(0, 1))
    assert s3.agent_pos == (1, 1)


def test_obstacle_blocks_without_key():
    """obstacle blocks movement when agent has no key."""
    env = GridEnvironment(
        width=3,
        height=1,
        obstacles=frozenset({(1, 0)}),
        key_pos=(2, 0),
        goal_pos=(2, 0),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    s2, _ = env.step(s, Move(1, 0))
    assert s2.agent_pos == (0, 0)


def test_key_lets_through_obstacle():
    """with key, agent can move through obstacles."""
    env = GridEnvironment(
        width=3,
        height=1,
        obstacles=frozenset({(2, 0)}),
        key_pos=(1, 0),
        goal_pos=(2, 0),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    s2, _ = env.step(s, Move(1, 0))
    assert s2.agent_pos == (1, 0)
    s3, _ = env.step(s2, PickupKey())
    assert s3.key_pos is None
    s4, _ = env.step(s3, Move(1, 0))
    assert s4.agent_pos == (2, 0)


def test_pickup_key():
    """pickup sets key_pos to None when on key cell."""
    env = GridEnvironment(
        width=2,
        height=1,
        obstacles=frozenset(),
        key_pos=(1, 0),
        goal_pos=(1, 0),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    s2, _ = env.step(s, Move(1, 0))
    s3, _ = env.step(s2, PickupKey())
    assert s3.key_pos is None


def test_terminal_at_goal():
    """is_terminal true when agent at goal."""
    env = GridEnvironment(
        width=2,
        height=1,
        obstacles=frozenset(),
        key_pos=(0, 0),
        goal_pos=(1, 0),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    assert not env.is_terminal(s)
    s2, _ = env.step(s, Move(1, 0))
    assert env.is_terminal(s2)


def test_available_actions():
    """available_actions returns valid moves and pickup when applicable."""
    env = GridEnvironment(
        width=2,
        height=2,
        obstacles=frozenset(),
        key_pos=(1, 0),
        goal_pos=(1, 1),
        agent_start=(0, 0),
    )
    s = env.initial_state()
    actions = env.available_actions(s)
    assert Move(1, 0) in actions
    assert Move(0, 1) in actions
    assert PickupKey() not in actions
    s2, _ = env.step(s, Move(1, 0))
    actions2 = env.available_actions(s2)
    assert PickupKey() in actions2
