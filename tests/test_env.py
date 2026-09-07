"""Behavioral checks for the simulator/evaluator boundary (standard unittest)."""

from dataclasses import FrozenInstanceError
from itertools import product
import unittest

import numpy as np

from yoked_plasticity.env import Context, World


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.world = World(73)
        self.anchor = Context(((0, 1, 0, 1), (1, 0, 1, 0)), (1.0, 0.2))

    def test_oracle_policy_attains_exact_root_value(self):
        for gamma in (0.0, 0.97, 1.0):
            world = World(73, gamma=gamma)
            for index in (0, 1):
                anchor = world.training_context(index)
                contexts = [anchor] + [
                    world.post_context(anchor, kind)
                    for kind in ("visible_refit", "hidden_opportunity", "mixed")
                ]
                for context in contexts:
                    with self.subTest(gamma=gamma, context=context):
                        q_values = world.oracle(context)
                        expected = gamma**4 * max(context.rewards)
                        self.assertAlmostEqual(float(q_values[0].max()), expected)
                        self.assertAlmostEqual(world.greedy_return(q_values, context), expected)
                        np.testing.assert_array_equal(q_values[9], [0.0, 0.0])

    def test_uniform_reference_matches_exhaustive_action_tapes(self):
        # Complete tapes include unused actions after early termination. Averaging
        # all 32 tapes therefore weights each observed prefix correctly.
        returns = []
        for actions in product((0, 1), repeat=5):
            state = 0
            result = 0.0
            for time, action in enumerate(actions):
                state, reward, terminated = self.world.step(state, action, self.anchor)
                result += self.world.gamma**time * reward
                if terminated:
                    break
            returns.append(result)
        self.assertAlmostEqual(self.world.random_return(self.anchor), sum(returns) / 32)

    def test_hidden_shift_preserves_entire_old_path(self):
        for index in (0, 1):
            anchor = self.world.training_context(index)
            hidden = self.world.post_context(anchor, "hidden_opportunity")
            best = index % 2
            self.assertEqual(hidden.codes, anchor.codes)
            self.assertEqual(hidden.rewards[best], anchor.rewards[best])
            self.assertEqual(hidden.rewards[1 - best], 1.5)
            old_state = new_state = 0
            for action in (best, *anchor.codes[best]):
                np.testing.assert_array_equal(
                    self.world.embeddings[old_state], self.world.embeddings[new_state]
                )
                old_transition = self.world.step(old_state, action, anchor)
                new_transition = self.world.step(new_state, action, hidden)
                self.assertEqual(old_transition, new_transition)
                old_state = old_transition[0]
                new_state = new_transition[0]
            self.assertEqual(old_transition, (9, 1.0, True))

    def test_matched_visible_and_mixed_switches(self):
        for index in (0, 1):
            anchor = self.world.training_context(index)
            best = index % 2
            visible = self.world.post_context(anchor, "visible_refit")
            mixed = self.world.post_context(anchor, "mixed")
            self.assertEqual(visible.rewards, anchor.rewards)
            self.assertEqual(visible.codes[best], tuple(1 - bit for bit in anchor.codes[best]))
            self.assertEqual(visible.codes[1 - best], anchor.codes[1 - best])
            self.assertEqual(mixed.codes, visible.codes)
            self.assertEqual(mixed.rewards[best], 0.2)
            self.assertEqual(mixed.rewards[1 - best], 1.5)
            # The formerly successful first action now immediately fails.
            self.assertEqual(
                self.world.step(1 + 4 * best, anchor.codes[best][0], visible),
                (9, -0.05, True),
            )

    def test_wrong_bits_terminate_at_every_depth(self):
        for branch in (0, 1):
            self.assertEqual(self.world.step(0, branch, self.anchor), (1 + 4 * branch, 0.0, False))
            for depth, correct in enumerate(self.anchor.codes[branch]):
                state = 1 + 4 * branch + depth
                self.assertEqual(self.world.step(state, 1 - correct, self.anchor), (9, -0.05, True))
                expected = (
                    (9, self.anchor.rewards[branch], True)
                    if depth == 3
                    else (state + 1, 0.0, False)
                )
                self.assertEqual(self.world.step(state, correct, self.anchor), expected)

    def test_context_never_changes_observations(self):
        snapshot = self.world.embeddings.copy()
        self.assertEqual(snapshot.dtype, np.dtype(np.float64))
        self.assertEqual(snapshot.shape, (10, 32))
        self.assertEqual(len({row.tobytes() for row in snapshot[:9]}), 9)
        self.assertTrue(np.all((snapshot[:9] == -1.0) | (snapshot[:9] == 1.0)))
        np.testing.assert_array_equal(snapshot[9], np.zeros(32))
        for index in (9, 0, 100, 1):
            anchor = self.world.training_context(index)
            for kind in ("visible_refit", "hidden_opportunity", "mixed"):
                context = self.world.post_context(anchor, kind)
                self.world.oracle(context)
                self.world.random_return(context)
                for state in range(9):
                    for action in (0, 1):
                        self.world.step(state, action, context)
            np.testing.assert_array_equal(self.world.embeddings, snapshot)
        np.testing.assert_array_equal(World(73).embeddings, snapshot)
        with self.assertRaises(ValueError):
            self.world.embeddings[0, 0] = 0.0
        with self.assertRaises(ValueError):
            self.world.embeddings.setflags(write=True)

    def test_context_generation_has_no_rng_consumption_dependency(self):
        expected = self.world.training_context(17)
        self.world.training_context(999)
        self.assertEqual(self.world.training_context(17), expected)
        self.assertEqual(World(73, observation_dim=64).training_context(17), expected)
        self.assertEqual(World(73).training_context(17), expected)
        self.assertEqual(self.world.training_context(0).rewards, (1.0, 0.2))
        self.assertEqual(self.world.training_context(1).rewards, (0.2, 1.0))

    def test_greedy_ties_choose_lowest_action(self):
        q_values = np.zeros((10, 2), dtype=np.float64)
        # All-zero actions choose branch 0, pass its first bit, then fail its
        # second bit. The root reward is at t=0, so this penalty is at t=2.
        self.assertAlmostEqual(
            self.world.greedy_return(q_values, self.anchor), -0.05 * self.world.gamma**2
        )

    def test_context_is_immutable_and_rejects_invalid_data(self):
        with self.assertRaises(FrozenInstanceError):
            self.anchor.rewards = (0.2, 1.0)
        bad_codes = (
            [(0, 1, 0, 1), (1, 0, 1, 0)],
            ((0, 1, 0), (1, 0, 1, 0)),
            ((0, 1, 2, 1), (1, 0, 1, 0)),
            ((0, 1, 0.0, 1), (1, 0, 1, 0)),
        )
        for codes in bad_codes:
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                Context(codes, (1.0, 0.2))
        for rewards in ([1.0, 0.2], (1.0,), (float("nan"), 0.2), (1.0, float("inf"))):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                Context(self.anchor.codes, rewards)

    def test_rejects_invalid_simulation_and_evaluation_inputs(self):
        for state in (-1, 9, 10, 0.5, True):
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.world.step(state, 0, self.anchor)
        for action in (-1, 2, 0.0, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.world.step(0, action, self.anchor)
        with self.assertRaises(TypeError):
            self.world.step(0, 0, None)
        with self.assertRaises(ValueError):
            self.world.post_context(self.anchor, "other")
        for index in (-1, 1.5, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.world.training_context(index)
        for values in (
            np.zeros((9, 2)),
            np.full((10, 2), np.nan),
            np.zeros((10, 2), dtype=np.complex128),
        ):
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                self.world.greedy_return(values, self.anchor)


if __name__ == "__main__":
    unittest.main()
