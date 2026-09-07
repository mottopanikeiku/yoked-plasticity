"""Small float64 Double DQN with externally scheduled target updates."""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np


class DQN:
    """Two ReLU hidden layers; all action selection and replay RNG are external.

    Parameters and Adam moments use [W1, b1, W2, b2, W3, b3] order.
    ``steps`` holds one Adam clock for each weight/bias layer pair.
    Target synchronization, when requested, happens after the optimizer step.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden: int = 64,
        seed: int = 0,
        learning_rate: float = 0.0003,
        gamma: float = 0.97,
    ) -> None:
        for name, value in (("input_dim", input_dim), ("hidden", hidden)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        rng = np.random.default_rng(seed)
        self.params = [
            rng.normal(0.0, math.sqrt(2.0 / self.input_dim), (self.input_dim, self.hidden)),
            np.zeros(self.hidden, dtype=np.float64),
            rng.normal(0.0, math.sqrt(2.0 / self.hidden), (self.hidden, self.hidden)),
            np.zeros(self.hidden, dtype=np.float64),
            rng.normal(0.0, 1.0 / math.sqrt(self.hidden), (self.hidden, 2)),
            np.zeros(2, dtype=np.float64),
        ]
        self.target_params = [p.copy() for p in self.params]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.steps = [0, 0, 0]
        self._base_online: list[np.ndarray] | None = None
        self._base_target: list[np.ndarray] | None = None
        self._frozen_reference: list[np.ndarray] | None = None
        # Operational counters, not learning state: count actual network/example
        # evaluations, including frozen branches, without perturbing hashes.
        self.forward_examples = 0
        self.frozen_forward_examples = 0

    def _observations(self, observations: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(observations, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2 or result.shape[1] != self.input_dim or result.shape[0] == 0:
            raise ValueError(f"{name} must have nonempty shape [batch, {self.input_dim}] or [{self.input_dim}]")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    def _forward(
        self, observations: np.ndarray, params: list[np.ndarray], *, frozen: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.forward_examples += observations.shape[0]
        if frozen:
            self.frozen_forward_examples += observations.shape[0]
        h1 = observations @ params[0]
        h1 += params[1]
        np.maximum(h1, 0.0, out=h1)
        h2 = h1 @ params[2]
        h2 += params[3]
        np.maximum(h2, 0.0, out=h2)
        q = h2 @ params[4]
        q += params[5]
        return h1, h2, q

    def _combine_frozen(
        self, observations: np.ndarray, q: np.ndarray, *, target: bool = False
    ) -> np.ndarray:
        if self._frozen_reference is None:
            return q
        base = self._base_target if target else self._base_online
        base_q = self._forward(observations, base, frozen=True)[2]
        reference_q = self._forward(observations, self._frozen_reference, frozen=True)[2]
        return base_q + (q - reference_q)

    def _values(self, observations: np.ndarray, *, target: bool = False) -> np.ndarray:
        params = self.target_params if target else self.params
        q = self._forward(observations, params)[2]
        return self._combine_frozen(observations, q, target=target)

    def q_values(self, observations: np.ndarray) -> np.ndarray:
        """Return [batch, 2] online Q values, including for a single observation."""
        observations = self._observations(observations, "observations")
        q = self._values(observations)
        if not np.isfinite(q).all():
            raise FloatingPointError("nonfinite online Q values")
        return q

    def _loss_and_gradients(
        self, observations: np.ndarray, actions: np.ndarray, targets: np.ndarray
    ) -> tuple[float, list[np.ndarray]]:
        """Mean Huber loss and gradients for validated arrays and detached targets.

        This helper does not bootstrap or mutate state, allowing finite-difference
        checks without differentiating through the Double-DQN target.
        """
        h1, h2, q = self._forward(observations, self.params)
        q = self._combine_frozen(observations, q)
        rows = np.arange(observations.shape[0])
        error = q[rows, actions] - targets
        absolute = np.abs(error)
        quadratic = np.minimum(absolute, 1.0)
        loss = float(np.mean(0.5 * quadratic * quadratic + absolute - quadratic))
        dq = np.zeros_like(q)
        dq[rows, actions] = np.clip(error, -1.0, 1.0) / observations.shape[0]
        dw3 = h2.T @ dq
        db3 = dq.sum(axis=0)
        dh2 = dq @ self.params[4].T
        dh2 *= h2 > 0.0
        dw2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ self.params[2].T
        dh1 *= h1 > 0.0
        dw1 = observations.T @ dh1
        db1 = dh1.sum(axis=0)
        return loss, [dw1, db1, dw2, db2, dw3, db3]

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
        """Take one Adam step on a mean Huber Double-DQN TD loss.

        Each transition vector has shape [batch]. Termination may be boolean or
        numeric zero/one. Only true termination masks bootstrap; callers must not
        mark a context boundary or a nonterminal time limit as termination.
        """
        observations = self._observations(observations, "observations")
        next_observations = self._observations(next_observations, "next_observations")
        if observations.shape != next_observations.shape:
            raise ValueError("observations and next_observations must have matching shapes")
        batch = observations.shape[0]
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

        # Do not even evaluate terminal next states: their values cannot affect
        # the loss, including when a finite embedding would overflow the MLP.
        alive = terminated == 0
        targets = rewards.copy()
        if np.any(alive):
            next_live = next_observations if np.all(alive) else next_observations[alive]
            online_q = self._values(next_live)
            target_q = self._values(next_live, target=True)
            if not np.isfinite(online_q).all() or not np.isfinite(target_q).all():
                raise FloatingPointError("nonfinite bootstrap Q values")
            next_actions = online_q.argmax(axis=1)
            targets[alive] += self.gamma * target_q[np.arange(next_actions.size), next_actions]
        loss, gradients = self._loss_and_gradients(observations, actions, targets)
        if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients):
            raise FloatingPointError("nonfinite TD loss or gradients")
        for layer in range(3):
            self.steps[layer] += 1
            t = self.steps[layer]
            correction1 = 1.0 - 0.9 ** t
            correction2 = 1.0 - 0.999 ** t
            for index in (2 * layer, 2 * layer + 1):
                gradient = gradients[index]
                moment = self.m[index]
                variance = self.v[index]
                moment *= 0.9
                moment += 0.1 * gradient
                variance *= 0.999
                variance += 0.001 * gradient * gradient
                self.params[index] -= self.learning_rate * (moment / correction1) / (np.sqrt(variance / correction2) + 1e-8)
        if sync_target:
            for target, online in zip(self.target_params, self.params):
                np.copyto(target, online)
            if self._base_online is not None:
                for target, online in zip(self._base_target, self._base_online):
                    np.copyto(target, online)
        return loss

    def clone(self) -> DQN:
        """Copy every learned/optimizer/target state, without consuming randomness."""
        result = object.__new__(type(self))
        result.input_dim = self.input_dim
        result.hidden = self.hidden
        result.learning_rate = self.learning_rate
        result.gamma = self.gamma
        for name in ("params", "target_params", "m", "v"):
            setattr(result, name, [array.copy() for array in getattr(self, name)])
        result.steps = self.steps.copy()
        for name in ("_base_online", "_base_target", "_frozen_reference"):
            group = getattr(self, name)
            setattr(result, name, None if group is None else [array.copy() for array in group])
        result.forward_examples = self.forward_examples
        result.frozen_forward_examples = self.frozen_forward_examples
        return result

    def reset_head(self, seed: int) -> None:
        """Replace both heads identically; reset only the final layer's Adam state."""
        rng = np.random.default_rng(seed)
        head = rng.normal(0.0, 1.0 / math.sqrt(self.hidden), (self.hidden, 2))
        np.copyto(self.params[4], head)
        self.params[5].fill(0.0)
        np.copyto(self.target_params[4], head)
        self.target_params[5].fill(0.0)
        for index in (4, 5):
            self.m[index].fill(0.0)
            self.v[index].fill(0.0)
        self.steps[2] = 0

    def inject(self, seed: int) -> None:
        """Freeze prior functions and train a fresh residual, initially zero.

        Online = old_online + (g - g0); target = old_target + (g_target - g0).
        Only g is trainable. Each value computation now evaluates three networks;
        forward_examples/frozen_forward_examples measure the actual extra work.
        Repeated injection is rejected rather than growing nested functions.
        """
        if self._base_online is not None:
            raise ValueError("plasticity injection is supported only once per learner")
        fresh = DQN(self.input_dim, self.hidden, seed, self.learning_rate, self.gamma)
        self._base_online = self.params
        self._base_target = self.target_params
        self.params = fresh.params
        self.target_params = fresh.target_params
        self._frozen_reference = [array.copy() for array in self.params]
        self.m = fresh.m
        self.v = fresh.v
        self.steps = fresh.steps

    def fingerprint(self) -> str:
        """Hash learning state, including frozen branches, but not work counters."""
        digest = hashlib.sha256(b"yoked-plasticity-dqn-v2\0")
        digest.update(struct.pack("<QQdd", self.input_dim, self.hidden, self.learning_rate, self.gamma))
        digest.update(bytes([self._base_online is not None]))
        groups = [self.params, self.target_params, self.m, self.v]
        if self._base_online is not None:
            groups.extend((self._base_online, self._base_target, self._frozen_reference))
        for group in groups:
            for array in group:
                digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
        for step in self.steps:
            digest.update(struct.pack("<Q", step))
        return digest.hexdigest()
