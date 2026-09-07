"""Numerical and state-boundary regressions for the temporary CPU learner."""

import unittest

import numpy as np

from yoked_plasticity.learner import DQN


class DQNTests(unittest.TestCase):
    @staticmethod
    def batch():
        rng = np.random.default_rng(902)
        observations = rng.normal(size=(7, 3))
        return (
            observations,
            np.array([0, 1, 1, 0, 1, 0, 1]),
            np.array([0.3, -0.5, 1.0, 0.7, -0.2, 0.4, 0.9]),
            rng.normal(size=(7, 3)),
            np.array([False, True, False, False, True, False, True]),
        )

    def test_gradients_match_finite_differences_on_both_huber_branches(self):
        learner = DQN(input_dim=3, hidden=4, seed=71)
        # Fixed activation regions include inactive neurons in both layers.
        learner.params[0] *= 0.03
        learner.params[1][:] = [0.5, -0.5, 0.6, -0.6]
        learner.params[2] *= 0.03
        learner.params[3][:] = [0.5, -0.5, 0.6, -0.6]
        observations = np.random.default_rng(82).normal(size=(4, 3))
        actions = np.array([0, 1, 1, 0])
        residuals = np.array([0.25, -0.4, 1.6, -2.1])
        targets = learner.q_values(observations)[np.arange(4), actions] - residuals
        loss, gradients = learner._loss_and_gradients(observations, actions, targets)
        self.assertAlmostEqual(loss, (0.5 * 0.25**2 + 0.5 * 0.4**2 + 1.1 + 1.6) / 4)
        epsilon = 1e-6
        for parameter, gradient in zip(learner.params, gradients):
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = learner._loss_and_gradients(observations, actions, targets)[0]
                parameter[index] = original - epsilon
                minus = learner._loss_and_gradients(observations, actions, targets)[0]
                parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-9)

    def test_double_dqn_selects_online_action_and_masks_termination(self):
        learner = DQN(input_dim=3, hidden=4, seed=7, gamma=0.5)
        for parameter in learner.params + learner.target_params:
            parameter.fill(0.0)
        learner.params[5][:] = [2.0, 1.0]
        learner.target_params[5][:] = [3.0, 20.0]
        target_before = [p.copy() for p in learner.target_params]
        loss = learner.update(
            np.zeros((2, 3)),
            np.array([0, 0]),
            np.array([0.25, 0.25]),
            np.zeros((2, 3)),
            np.array([True, False]),
        )
        # Terminal target=.25; nonterminal target=.25+.5*3, not .25+.5*20.
        self.assertAlmostEqual(loss, (1.25 + 0.5 * 0.25**2) / 2)
        for old, current in zip(target_before, learner.target_params):
            np.testing.assert_array_equal(old, current)

    def test_terminal_next_values_cannot_affect_loss_or_state(self):
        original = DQN(input_dim=3, hidden=4, seed=9)
        changed_next = original.clone()
        observations, actions, rewards, next_observations, _ = self.batch()
        terminated = np.ones(len(actions), dtype=bool)
        loss = original.update(observations, actions, rewards, next_observations, terminated)
        # A terminal embedding is finite but its network evaluation can overflow.
        huge_next = np.full_like(next_observations, np.finfo(np.float64).max)
        with np.errstate(over="raise", invalid="raise"):
            other_loss = changed_next.update(observations, actions, rewards, huge_next, terminated)
        self.assertEqual(loss, other_loss)
        self.assertEqual(original.fingerprint(), changed_next.fingerprint())

    def test_clone_has_independent_exact_continuation(self):
        learner = DQN(input_dim=3, hidden=4, seed=12)
        batch = self.batch()
        learner.update(*batch, sync_target=True)
        learner.update(*batch)
        clone = learner.clone()
        self.assertEqual(learner.fingerprint(), clone.fingerprint())
        for name in ("params", "target_params", "m", "v"):
            for left, right in zip(getattr(learner, name), getattr(clone, name)):
                self.assertFalse(np.shares_memory(left, right))
        before = learner.fingerprint()
        clone_loss = clone.update(*batch, sync_target=True)
        self.assertEqual(before, learner.fingerprint())
        original_loss = learner.update(*batch, sync_target=True)
        self.assertEqual(original_loss, clone_loss)
        self.assertEqual(learner.fingerprint(), clone.fingerprint())
        for online, target in zip(learner.params, learner.target_params):
            np.testing.assert_array_equal(online, target)

    def test_reset_preserves_hidden_state_and_restarts_only_head_adam(self):
        learner = DQN(input_dim=3, hidden=4, seed=13)
        batch = self.batch()
        for _ in range(3):
            learner.update(*batch)
        before = learner.clone()
        twin = learner.clone()
        learner.reset_head(31)
        twin.reset_head(31)
        self.assertEqual(learner.fingerprint(), twin.fingerprint())
        self.assertNotEqual(before.fingerprint(), learner.fingerprint())
        for name in ("params", "target_params", "m", "v"):
            for prior, current in zip(getattr(before, name)[:4], getattr(learner, name)[:4]):
                np.testing.assert_array_equal(prior, current)
        for index in (4, 5):
            np.testing.assert_array_equal(learner.params[index], learner.target_params[index])
            np.testing.assert_array_equal(learner.m[index], np.zeros_like(learner.m[index]))
            np.testing.assert_array_equal(learner.v[index], np.zeros_like(learner.v[index]))
        np.testing.assert_array_equal(learner.params[5], np.zeros(2))
        self.assertEqual(learner.steps, [3, 3, 0])

        observations, actions, rewards, next_observations, _ = batch
        # Terminal targets give an independent, fixed-target first-step formula.
        _, gradients = learner._loss_and_gradients(observations, actions, rewards)
        expected_heads = [
            learner.params[index] - learner.learning_rate * gradients[index] / (np.abs(gradients[index]) + 1e-8)
            for index in (4, 5)
        ]
        learner.update(observations, actions, rewards, next_observations, np.ones(len(actions), dtype=bool))
        self.assertEqual(learner.steps, [4, 4, 1])
        for expected, actual in zip(expected_heads, learner.params[4:]):
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_fingerprint_covers_targets_moments_and_each_clock(self):
        learner = DQN(input_dim=3, hidden=4, seed=14)
        learner.update(*self.batch())
        digest = learner.fingerprint()
        for name in ("params", "target_params", "m", "v"):
            for index in range(6):
                clone = learner.clone()
                getattr(clone, name)[index].flat[0] += 0.125
                self.assertNotEqual(digest, clone.fingerprint(), (name, index))
        for index in range(3):
            clone = learner.clone()
            clone.steps[index] += 1
            self.assertNotEqual(digest, clone.fingerprint(), index)

    def test_public_shape_and_finite_boundaries(self):
        learner = DQN(seed=15)
        single = np.linspace(-1, 1, 32)
        self.assertEqual(learner.q_values(single).shape, (1, 2))
        np.testing.assert_array_equal(learner.q_values(single), learner.q_values(single[None, :]))
        self.assertEqual(learner.q_values(np.zeros((3, 32))).dtype, np.float64)
        for invalid in (np.zeros(31), np.zeros((0, 32)), np.zeros((1, 1, 32)), np.full(32, np.nan)):
            with self.assertRaises(ValueError):
                learner.q_values(invalid)
        learner = DQN(input_dim=3, hidden=4, seed=16)
        for field, invalid in (
            (1, np.ones(7, dtype=float)),
            (1, np.full(7, 2)),
            (2, np.full(7, np.inf)),
            (2, np.zeros((7, 1))),
            (3, np.zeros((6, 3))),
            (4, np.full(7, 0.5)),
            (4, np.full(7, np.nan)),
        ):
            batch = list(self.batch())
            batch[field] = invalid
            before = learner.fingerprint()
            with self.assertRaises(ValueError):
                learner.update(*batch)
            self.assertEqual(before, learner.fingerprint())

    def test_injection_preserves_both_functions_then_learns_and_syncs(self):
        learner = DQN(input_dim=3, hidden=4, seed=18)
        batch = self.batch()
        learner.update(*batch)
        observations = batch[0]
        online_before = learner.q_values(observations)
        target_before = learner._values(observations, target=True)
        self.assertFalse(np.array_equal(online_before, target_before))
        trainable_count = sum(p.size for p in learner.params)
        learner.inject(29)
        np.testing.assert_array_equal(online_before, learner.q_values(observations))
        np.testing.assert_array_equal(target_before, learner._values(observations, target=True))
        self.assertEqual(trainable_count, sum(p.size for p in learner.params))
        self.assertEqual(learner.steps, [0, 0, 0])
        for moment in learner.m + learner.v:
            np.testing.assert_array_equal(moment, np.zeros_like(moment))
        clone = learner.clone()
        initial_hash = learner.fingerprint()
        forward_before = learner.forward_examples
        frozen_before = learner.frozen_forward_examples
        learner.q_values(observations)
        self.assertEqual(learner.forward_examples - forward_before, 3 * len(observations))
        self.assertEqual(learner.frozen_forward_examples - frozen_before, 2 * len(observations))
        self.assertEqual(initial_hash, learner.fingerprint())
        frozen_before_update = [
            [p.copy() for p in group]
            for group in (learner._base_online, learner._base_target, learner._frozen_reference)
        ]
        loss = learner.update(*batch)
        self.assertEqual(initial_hash, clone.fingerprint())
        self.assertFalse(np.array_equal(online_before, learner.q_values(observations)))
        np.testing.assert_array_equal(target_before, learner._values(observations, target=True))
        for before_group, after_group in zip(
            frozen_before_update,
            (learner._base_online, learner._base_target, learner._frozen_reference),
        ):
            for before, after in zip(before_group, after_group):
                np.testing.assert_array_equal(before, after)
        self.assertEqual(loss, clone.update(*batch))
        self.assertEqual(learner.fingerprint(), clone.fingerprint())
        learner.update(*batch, sync_target=True)
        np.testing.assert_array_equal(
            learner.q_values(observations), learner._values(observations, target=True)
        )
        before_repeat = learner.fingerprint()
        with self.assertRaises(ValueError):
            learner.inject(33)
        self.assertEqual(before_repeat, learner.fingerprint())

    def test_injected_gradients_use_total_output_and_only_trainable_branch(self):
        learner = DQN(input_dim=3, hidden=4, seed=41)
        learner.update(*self.batch())
        learner.inject(42)
        learner.params[0] *= 0.03
        learner.params[1][:] = [0.5, -0.5, 0.6, -0.6]
        learner.params[2] *= 0.03
        learner.params[3][:] = [0.5, -0.5, 0.6, -0.6]
        observations, actions, _, _, _ = self.batch()
        targets = learner.q_values(observations)[np.arange(len(actions)), actions] - 0.3
        loss, gradients = learner._loss_and_gradients(observations, actions, targets)
        self.assertAlmostEqual(loss, 0.5 * 0.3**2)
        epsilon = 1e-6
        for parameter, gradient in zip(learner.params, gradients):
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = learner._loss_and_gradients(observations, actions, targets)[0]
                parameter[index] = original - epsilon
                minus = learner._loss_and_gradients(observations, actions, targets)[0]
                parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-9)
        original_hash = learner.fingerprint()
        for name in ("_base_online", "_base_target", "_frozen_reference"):
            for index in range(6):
                clone = learner.clone()
                self.assertFalse(np.shares_memory(getattr(learner, name)[index], getattr(clone, name)[index]))
                getattr(clone, name)[index].flat[0] += 0.125
                self.assertNotEqual(original_hash, clone.fingerprint())
                self.assertEqual(original_hash, learner.fingerprint())


if __name__ == "__main__":
    unittest.main()
