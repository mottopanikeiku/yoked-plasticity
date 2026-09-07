"""Finite branch simulator and privileged exact evaluators.

Only ``embeddings[state]`` is a learner input. Contexts, state IDs, and the
oracle belong to the simulator/evaluator; none is encoded in observations.
The caller owns episode boundaries, context schedules, and interaction counts.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True, slots=True)
class Context:
    codes: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    rewards: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.codes, tuple) or len(self.codes) != 2:
            raise ValueError("codes must be a tuple of two four-bit tuples")
        for code in self.codes:
            if not isinstance(code, tuple) or len(code) != 4:
                raise ValueError("each branch code must be a four-bit tuple")
            if any(
                isinstance(bit, (bool, np.bool_))
                or not isinstance(bit, Integral)
                or bit not in (0, 1)
                for bit in code
            ):
                raise ValueError("code bits must be integers 0 or 1")
        if not isinstance(self.rewards, tuple) or len(self.rewards) != 2:
            raise ValueError("rewards must be a tuple of two finite numbers")
        if any(
            isinstance(reward, (bool, np.bool_))
            or not isinstance(reward, Real)
            or not np.isfinite(reward)
            for reward in self.rewards
        ):
            raise ValueError("rewards must be finite real numbers")


def _integer(value: int, name: str, lower: int, upper: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is out of range")
    return int(value)


def _context(context: Context) -> None:
    if not isinstance(context, Context):
        raise TypeError("context must be a validated Context")


class World:
    """Deterministic five-decision episodes with fixed context-free observations.

    State 0 is the root, states 1..4 and 5..8 are the two branches, and
    state 9 is terminal. Stepping terminal is an error, not an interaction.
    Observation and context RNG streams are separate and stateless with
    respect to context lookup order.
    """

    def __init__(self, seed: int, observation_dim: int = 32, gamma: float = 0.97):
        self._seed = _integer(seed, "seed", 0)
        observation_dim = _integer(observation_dim, "observation_dim", 4)
        if (
            isinstance(gamma, (bool, np.bool_))
            or not isinstance(gamma, Real)
            or not np.isfinite(gamma)
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError("gamma must be finite and in [0, 1]")
        self.gamma = float(gamma)
        rng = np.random.default_rng(np.random.SeedSequence(self._seed, spawn_key=(0,)))
        observations = np.zeros((10, observation_dim), dtype=np.float64)
        seen: set[bytes] = set()
        for state in range(9):
            while True:
                row = rng.integers(0, 2, size=observation_dim).astype(np.float64)
                row *= 2.0
                row -= 1.0
                key = row.tobytes()
                if key not in seen:
                    seen.add(key)
                    observations[state] = row
                    break
        # Immutable backing prevents callers from changing shared observations,
        # including by re-enabling the NumPy WRITEABLE flag.
        self._embeddings = np.frombuffer(observations.tobytes(), dtype=np.float64).reshape(
            10, observation_dim
        )

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def training_context(self, index: int) -> Context:
        index = _integer(index, "context index", 0)
        rng = np.random.default_rng(
            np.random.SeedSequence(self._seed, spawn_key=(1, index))
        )
        bits = rng.integers(0, 2, size=(2, 4))
        codes = (tuple(map(int, bits[0])), tuple(map(int, bits[1])))
        rewards = (1.0, 0.2) if index % 2 == 0 else (0.2, 1.0)
        return Context(codes=codes, rewards=rewards)

    def post_context(self, anchor: Context, kind: str) -> Context:
        _context(anchor)
        if kind not in ("visible_refit", "hidden_opportunity", "mixed"):
            raise ValueError(f"unknown context switch: {kind!r}")
        best = 0 if anchor.rewards[0] >= anchor.rewards[1] else 1
        other = 1 - best
        codes = list(anchor.codes)
        rewards = list(anchor.rewards)
        if kind in ("visible_refit", "mixed"):
            codes[best] = tuple(1 - bit for bit in codes[best])
        if kind in ("hidden_opportunity", "mixed"):
            rewards[other] = 1.5
        if kind == "mixed":
            rewards[best] = 0.2
        return Context(codes=tuple(codes), rewards=tuple(rewards))

    def step(self, state: int, action: int, context: Context) -> tuple[int, float, bool]:
        state = _integer(state, "nonterminal state", 0, 8)
        action = _integer(action, "action", 0, 1)
        _context(context)
        if state == 0:
            return 1 + 4 * action, 0.0, False
        branch, depth = divmod(state - 1, 4)
        if action != context.codes[branch][depth]:
            return 9, -0.05, True
        if depth == 3:
            return 9, float(context.rewards[branch]), True
        return state + 1, 0.0, False

    def oracle(self, context: Context) -> np.ndarray:
        """Privileged optimal Q by backward induction, including zero terminal Q."""
        _context(context)
        values = np.zeros((10, 2), dtype=np.float64)
        for state in range(8, -1, -1):
            for action in (0, 1):
                next_state, reward, terminated = self.step(state, action, context)
                values[state, action] = reward
                if not terminated:
                    values[state, action] += self.gamma * np.max(values[next_state])
        return values

    def random_return(self, context: Context) -> float:
        """Exact discounted root return for independent uniform binary actions."""
        _context(context)
        values = np.zeros(10, dtype=np.float64)
        for state in range(8, -1, -1):
            for action in (0, 1):
                next_state, reward, terminated = self.step(state, action, context)
                continuation = 0.0 if terminated else self.gamma * values[next_state]
                values[state] += 0.5 * (reward + continuation)
        return float(values[0])

    def greedy_return(self, q_values: np.ndarray, context: Context) -> float:
        """Evaluate a supplied policy; ties choose action 0, never consult oracle."""
        _context(context)
        q_values = np.asarray(q_values)
        if (
            q_values.shape != (10, 2)
            or not np.issubdtype(q_values.dtype, np.number)
            or not np.isrealobj(q_values)
            or not np.all(np.isfinite(q_values))
        ):
            raise ValueError("q_values must be a finite real array of shape (10, 2)")
        state = 0
        discount = 1.0
        result = 0.0
        while True:
            action = int(np.argmax(q_values[state]))
            state, reward, terminated = self.step(state, action, context)
            result += discount * reward
            if terminated:
                return float(result)
            discount *= self.gamma
