"""grid environment: obstacles, key, goal."""

from dataclasses import dataclass
from typing import Any

from track.environment import Action, Environment, State


@dataclass(frozen=True)
class GridState(State):
    """immutable grid state. key_pos is None when agent holds key."""

    width: int
    height: int
    agent_pos: tuple[int, int]
    key_pos: tuple[int, int] | None
    goal_pos: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]


class GridAction(Action):
    """base type for grid actions."""

    pass


@dataclass(frozen=True)
class Move(GridAction):
    """move by (dx, dy). dx, dy in {-1, 0, 1}."""

    dx: int
    dy: int


@dataclass(frozen=True)
class PickupKey(GridAction):
    """pick up key when standing on key cell."""

    pass


class GridEnvironment(Environment):
    """2d grid with obstacles, key, goal. obstacles block unless agent has key."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: frozenset[tuple[int, int]],
        key_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        agent_start: tuple[int, int],
    ) -> None:
        """build grid. all positions as (x, y). obstacles block unless agent has key."""
        self._width = width
        self._height = height
        self._obstacles = obstacles
        self._key_pos = key_pos
        self._goal_pos = goal_pos
        self._agent_start = agent_start

    def initial_state(self) -> GridState:
        """return initial state with agent at agent_start, key on key_pos."""
        return GridState(
            width=self._width,
            height=self._height,
            agent_pos=self._agent_start,
            key_pos=self._key_pos,
            goal_pos=self._goal_pos,
            obstacles=self._obstacles,
        )

    def available_actions(self, state: GridState) -> list[Action]:
        """return valid moves and pickup if on key cell."""
        actions: list[Action] = []
        ax, ay = state.agent_pos
        has_key = state.key_pos is None

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < state.width and 0 <= ny < state.height:
                target = (nx, ny)
                # obstacles block without key
                if target in state.obstacles and not has_key:
                    continue
                actions.append(Move(dx, dy))

        # pickup only when standing on key cell
        if state.key_pos is not None and state.agent_pos == state.key_pos:
            actions.append(PickupKey())

        return actions

    def step(self, state: GridState, action: Action) -> tuple[GridState, dict[str, Any]]:
        """apply action. invalid actions return same state."""
        if isinstance(action, Move):
            return self._step_move(state, action)
        if isinstance(action, PickupKey):
            return self._step_pickup(state)
        return state, {}

    def _step_move(self, state: GridState, action: Move) -> tuple[GridState, dict[str, Any]]:
        ax, ay = state.agent_pos
        nx, ny = ax + action.dx, ay + action.dy
        # out of bounds: no-op so runner doesn't crash on invalid actions
        if nx < 0 or nx >= state.width or ny < 0 or ny >= state.height:
            return state, {}
        target = (nx, ny)
        has_key = state.key_pos is None
        # blocked by obstacle without key
        if target in state.obstacles and not has_key:
            return state, {}
        new_state = GridState(
            width=state.width,
            height=state.height,
            agent_pos=(nx, ny),
            key_pos=state.key_pos,
            goal_pos=state.goal_pos,
            obstacles=state.obstacles,
        )
        return new_state, {}

    def _step_pickup(self, state: GridState) -> tuple[GridState, dict[str, Any]]:
        # not on key cell: no-op
        if state.key_pos is None or state.agent_pos != state.key_pos:
            return state, {}
        new_state = GridState(
            width=state.width,
            height=state.height,
            agent_pos=state.agent_pos,
            key_pos=None,
            goal_pos=state.goal_pos,
            obstacles=state.obstacles,
        )
        return new_state, {}

    def is_terminal(self, state: GridState) -> bool:
        """true when agent reached goal."""
        return state.agent_pos == state.goal_pos
