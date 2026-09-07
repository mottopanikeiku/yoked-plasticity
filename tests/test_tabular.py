"""Observable learning and state-isolation contracts for the tabular comparator."""

import unittest

import numpy as np

from yoked_plasticity.tabular import TabularQ


class TabularQTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = np.eye(3, dtype=np.float64)

    def test_terminal_mask_excludes_learned_next_state_values(self):
        learner = TabularQ(self.embeddings, learning_rate=1.0, gamma=0.5)
        learner.update(self.embeddings[1], [1], [100.0], self.embeddings[2], [True])
        live = learner.clone()
        terminal_loss = learner.update(self.embeddings[0], [1], [2.0], self.embeddings[1], [True])
        live.update(self.embeddings[0], [1], [2.0], self.embeddings[1], [False])
        self.assertEqual(learner.q_values(self.embeddings[0])[0, 1], 2.0)
        self.assertEqual(live.q_values(self.embeddings[0])[0, 1], 52.0)
        self.assertEqual(terminal_loss, 1.5)

    def test_duplicate_samples_use_sequential_current_table_bootstrap(self):
        learner = TabularQ(self.embeddings, learning_rate=0.5, gamma=0.5)
        observations = self.embeddings[[0, 0]]
        loss = learner.update(observations, [1, 1], [1.0, 1.0], observations, [False, False])
        # First: .5 * (1 - 0) = .5. Second: .5 + .5 * (1 + .5*.5 - .5) = .875.
        np.testing.assert_array_equal(learner.q_values(self.embeddings[0]), [[0.0, 0.875]])
        self.assertEqual(loss, (0.5 + 0.5 * 0.75 ** 2) / 2)
        self.assertEqual(learner.forward_examples, 0)
        self.assertEqual(learner.frozen_forward_examples, 0)

    def test_short_chain_learning_changes_greedy_return(self):
        learner = TabularQ(self.embeddings, learning_rate=0.5, gamma=0.8)

        def greedy_return():
            policy = learner.q_values(self.embeddings).argmax(axis=1)
            if policy[0] == 0:
                return 0.25
            return learner.gamma * (2.0 if policy[1] == 1 else 0.0)

        self.assertEqual(greedy_return(), 0.25)
        # Both actions are observed at each nonterminal state. The rewarding
        # chain is root --1--> middle --1--> terminal; root action 0 exits early.
        for _ in range(40):
            learner.update(
                self.embeddings[[0, 1, 0, 1]],
                [1, 1, 0, 0],
                [0.0, 2.0, 0.25, 0.0],
                self.embeddings[[1, 2, 2, 2]],
                [False, True, True, True],
            )
        self.assertEqual(greedy_return(), 1.6)
        np.testing.assert_array_equal(learner.q_values(self.embeddings[:2]).argmax(axis=1), [1, 1])
        np.testing.assert_allclose(learner.q_values(self.embeddings[:2]), [[0.25, 1.6], [0.0, 2.0]], atol=1e-8, rtol=0)

    def test_clone_is_independent_and_continues_exactly(self):
        learner = TabularQ(self.embeddings, learning_rate=0.3, gamma=0.7)
        batch = (self.embeddings[[1, 0]], [1, 1], [2.0, 0.0], self.embeddings[[2, 1]], [True, False])
        learner.update(*batch)
        clone = learner.clone()
        self.assertEqual(learner.fingerprint(), clone.fingerprint())
        before_values = learner.q_values(self.embeddings)
        before_hash = learner.fingerprint()
        clone_loss = clone.update(*batch, sync_target=True)
        np.testing.assert_array_equal(learner.q_values(self.embeddings), before_values)
        self.assertEqual(learner.fingerprint(), before_hash)
        self.assertNotEqual(clone.fingerprint(), before_hash)
        # Synchronization must not alter Q-learning: there is no target network.
        self.assertEqual(learner.update(*batch, sync_target=False), clone_loss)
        self.assertEqual(learner.fingerprint(), clone.fingerprint())

    def test_vocabulary_is_exact_and_independent_of_constructor_input(self):
        original = self.embeddings.copy()
        learner = TabularQ(self.embeddings)
        self.embeddings[0, 0] = 2.0
        learner.update(original[0], [1], [1.0], original[2], [True])
        np.testing.assert_array_equal(learner.q_values(original[0]), [[0.0, 0.1]])
        almost_known = original[0].copy()
        almost_known[0] = np.nextafter(almost_known[0], np.inf)
        for unknown in (almost_known, self.embeddings[0]):
            with self.assertRaises(ValueError):
                learner.q_values(unknown)
        # Consumers cannot modify learning state through returned Q values.
        returned = learner.q_values(original)
        returned[:] = 99.0
        np.testing.assert_array_equal(learner.q_values(original[0]), [[0.0, 0.1]])

    def test_fingerprint_binds_learning_semantics_but_not_read_counters(self):
        learner = TabularQ(self.embeddings)
        digest = learner.fingerprint()
        learner.q_values(self.embeddings)
        self.assertEqual(digest, learner.fingerprint())
        for other in (
            TabularQ(self.embeddings, gamma=0.5),
            TabularQ(self.embeddings, learning_rate=0.2),
            TabularQ(self.embeddings[::-1]),
            TabularQ(self.embeddings + 1.0),
        ):
            self.assertNotEqual(digest, other.fingerprint())
        learner.update(self.embeddings[0], [1], [1.0], self.embeddings[2], [True])
        self.assertNotEqual(digest, learner.fingerprint())

    def test_malformed_batches_are_rejected_before_any_transition_updates(self):
        learner = TabularQ(self.embeddings)
        valid = (self.embeddings[[0, 1]], [1, 0], [1.0, 2.0], self.embeddings[[1, 2]], [False, True])
        unknown_next = self.embeddings[[1, 2]].copy()
        unknown_next[1, 0] = 0.1
        for field, invalid in (
            (0, np.zeros((0, 3))),
            (0, np.zeros((2, 4))),
            (0, np.full((2, 3), np.nan)),
            (1, [0.0, 1.0]),
            (1, [0, 2]),
            (1, [False, True]),
            (2, [1.0, np.inf]),
            (2, [[1.0], [2.0]]),
            (3, self.embeddings[:1]),
            (3, unknown_next),
            (4, [0.0, 0.5]),
            (4, [False]),
        ):
            with self.subTest(field=field, invalid=invalid):
                batch = list(valid)
                batch[field] = invalid
                before = learner.fingerprint()
                with self.assertRaises(ValueError):
                    learner.update(*batch)
                self.assertEqual(learner.fingerprint(), before)
        with self.assertRaises(ValueError):
            learner.update(*valid, sync_target=1)

    def test_invalid_vocabulary_and_hyperparameters_are_rejected(self):
        for embeddings in (np.zeros((0, 3)), np.zeros((3, 0)), np.zeros(3), np.zeros((2, 3)), np.full((2, 3), np.inf)):
            with self.subTest(embeddings=embeddings), self.assertRaises(ValueError):
                TabularQ(embeddings)
        for kwargs in (
            {"learning_rate": 0.0},
            {"learning_rate": np.inf},
            {"learning_rate": True},
            {"gamma": -0.1},
            {"gamma": 1.1},
            {"gamma": np.nan},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TabularQ(self.embeddings, **kwargs)

    def test_arithmetic_overflow_cannot_store_nonfinite_q_values(self):
        learner = TabularQ(self.embeddings, learning_rate=np.finfo(np.float64).max)
        before = learner.fingerprint()
        with self.assertRaises(FloatingPointError):
            learner.update(self.embeddings[0], [1], [2.0], self.embeddings[2], [True])
        self.assertEqual(learner.fingerprint(), before)
        np.testing.assert_array_equal(learner.q_values(self.embeddings), np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
