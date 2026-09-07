"""Representation-favorable non-neural Q-learning calibration comparator.

Only exact observable embedding bytes identify table rows. There is no context,
transition-model, reward-model, or oracle access, and no nearest-neighbor lookup.
Knowing the finite observation vocabulary removes neural representation learning;
this comparator is not evidence that neural plasticity exists or is impaired.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
import struct
from types import MappingProxyType

import numpy as np


class TabularQ:
    """Binary-action Q-learning over a fixed, exact observation vocabulary.

    Inputs are converted to little-endian float64 before byte lookup. Distinct
    vocabulary rows must have distinct bytes; unknown observations are errors,
    never assigned a guessed state. The vocabulary is copied into immutable byte
    keys, so subsequent changes to the constructor input cannot change identity.
    """

    def __init__(self, embeddings: np.ndarray, learning_rate: float = 0.1, gamma: float = 0.97) -> None:
        for name, value in (("learning_rate", learning_rate), ("gamma", gamma)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        embeddings = np.asarray(embeddings, dtype="<f8")
        if embeddings.ndim != 2 or 0 in embeddings.shape or not np.isfinite(embeddings).all():
            raise ValueError("embeddings must have finite nonempty shape [states, observation_dim]")
        keys = tuple(row.tobytes() for row in embeddings)
        state_ids = {key: index for index, key in enumerate(keys)}
        if len(state_ids) != len(keys):
            raise ValueError("embeddings must have unique observation bytes")
        self.input_dim = embeddings.shape[1]
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self._embedding_keys = keys
        self._state_ids = MappingProxyType(state_ids)
        self.q_table = np.zeros((len(keys), 2), dtype=np.float64)
        # Operational counters are excluded from learning-state fingerprints.
        # No neural forward occurs, including during bootstrap or evaluation.
        self.forward_examples = 0
        self.frozen_forward_examples = 0
        self.table_lookups = 0

    def _states(self, observations: np.ndarray, name: str) -> np.ndarray:
        observations = np.asarray(observations, dtype="<f8")
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.ndim != 2 or observations.shape[1] != self.input_dim or observations.shape[0] == 0:
            raise ValueError(f"{name} must have nonempty shape [batch, {self.input_dim}] or [{self.input_dim}]")
        if not np.isfinite(observations).all():
            raise ValueError(f"{name} must be finite")
        states = np.empty(observations.shape[0], dtype=np.int64)
        for index, row in enumerate(observations):
            try:
                states[index] = self._state_ids[row.tobytes()]
            except KeyError:
                raise ValueError(f"{name} contains an unknown observation") from None
        return states

    def q_values(self, observations: np.ndarray) -> np.ndarray:
        """Return independent [batch, 2] values, also accepting one observation."""
        states = self._states(observations, "observations")
        values = self.q_table[states]
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite Q table values")
        self.table_lookups += states.size
        return values

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminated: np.ndarray,
        *,
        sync_target: bool = False,
    ) -> float:
        """Apply every transition in order using the current-table max bootstrap.

        Repeated state/action samples therefore update repeatedly, and earlier
        samples can change the bootstrap of later samples in the same batch.
        True termination alone masks bootstrap. ``sync_target`` is validated but
        ignored: ordinary Q-learning has no target network. The returned mean
        Huber TD error is diagnostic; the update itself is alpha * TD error,
        without Huber clipping. Malformed batches are rejected before learning;
        an arithmetic failure stops before its transition, not earlier ones.
        """
        states = self._states(observations, "observations")
        next_states = self._states(next_observations, "next_observations")
        if states.shape != next_states.shape:
            raise ValueError("observations and next_observations must have matching shapes")
        batch = states.size
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float64)
        terminated = np.asarray(terminated)
        if actions.shape != (batch,) or actions.dtype.kind not in "iu" or np.any((actions < 0) | (actions > 1)):
            raise ValueError("actions must be an integer vector of shape [batch] with values 0 or 1")
        if rewards.shape != (batch,) or not np.isfinite(rewards).all():
            raise ValueError("rewards must be a finite vector of shape [batch]")
        if terminated.shape != (batch,) or terminated.dtype.kind not in "biuf" or not np.all((terminated == 0) | (terminated == 1)):
            raise ValueError("terminated must be a boolean or zero/one vector of shape [batch]")
        if not isinstance(sync_target, (bool, np.bool_)):
            raise ValueError("sync_target must be boolean")

        mean_loss = 0.0
        for index in range(batch):
            state, action = states[index], actions[index]
            previous = float(self.q_table[state, action])
            target = float(rewards[index])
            if not terminated[index]:
                next_values = self.q_table[next_states[index]]
                if not np.isfinite(next_values).all():
                    raise FloatingPointError("nonfinite bootstrap Q values")
                target += self.gamma * float(max(next_values[0], next_values[1]))
            error = target - previous
            updated = previous + self.learning_rate * error
            if not all(math.isfinite(value) for value in (target, error, updated)):
                raise FloatingPointError("nonfinite Q-learning update")
            magnitude = abs(error)
            loss = 0.5 * error * error if magnitude <= 1.0 else magnitude - 0.5
            mean_loss += (loss - mean_loss) / (index + 1)
            self.q_table[state, action] = updated
            self.table_lookups += 1 + int(not terminated[index])
        return mean_loss

    def clone(self) -> TabularQ:
        """Copy table and counters exactly; only immutable vocabulary is shared."""
        result = object.__new__(type(self))
        result.input_dim = self.input_dim
        result.learning_rate = self.learning_rate
        result.gamma = self.gamma
        result._embedding_keys = self._embedding_keys
        result._state_ids = self._state_ids
        result.q_table = self.q_table.copy()
        result.forward_examples = self.forward_examples
        result.frozen_forward_examples = self.frozen_forward_examples
        result.table_lookups = self.table_lookups
        return result

    def fingerprint(self) -> str:
        """Bind vocabulary-to-row mapping, table, gamma, and rate, not counters."""
        digest = hashlib.sha256(b"yoked-plasticity-tabular-q-v1\0")
        digest.update(struct.pack("<QQdd", len(self._embedding_keys), self.input_dim, self.learning_rate, self.gamma))
        for key in self._embedding_keys:
            digest.update(key)
        digest.update(np.asarray(self.q_table, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()
