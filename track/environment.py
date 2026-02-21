"""abstract environment interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Action:
    """base type for actions. subclasses define concrete actions."""

    pass


@dataclass(frozen=True)
class State:
    """immutable environment state. structure depends on environment."""

    pass


class Environment(ABC):
    """abstract base for environments. exposes states, actions, transitions."""

    @abstractmethod
    def initial_state(self) -> State:
        """return the initial state of the environment."""
        pass

    @abstractmethod
    def available_actions(self, state: State) -> list[Action]:
        """return actions valid in this state (regardless of agent rules)."""
        pass

    @abstractmethod
    def step(self, state: State, action: Action) -> tuple[State, dict[str, Any]]:
        """apply action to state. return (new_state, info_dict)."""
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """true if episode is done (e.g., goal reached)."""
        pass
