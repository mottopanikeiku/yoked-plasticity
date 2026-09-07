"""Integrity boundaries whose failure would invalidate the causal cells."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from yoked_plasticity.env import World
from yoked_plasticity.experiment import Replay, age, consume, generate, load_checkpoint, run_pair, save_checkpoint


def config():
    return dict(observation_dim=8, hidden=8, learning_rate=0.001, gamma=0.97,
                replay_capacity=16, batch_size=4, epsilon=0.1, update_every=1,
                target_every=4, age_steps=64, block_steps=32, adapt_steps=64,
                evaluate_every=16, restart_steps=16, recent_keep=8,
                switches=["mixed"], intervention="identity")


class ReplayIntegrityTests(unittest.TestCase):
    def test_overwritten_uid_cannot_silently_read_a_new_transition(self):
        replay = Replay(4)
        world = World(9, observation_dim=8)
        for uid in range(6):
            replay.append((0, 0, float(uid), 1, False))
        with self.assertRaises(ValueError):
            replay.batch(np.array([0]), world.embeddings)
        np.testing.assert_array_equal(replay.batch(np.array([2, 5]), world.embeddings)[2], [2, 5])
        recent = replay.recent(2)
        recent.append((0, 0, 6.0, 1, False))
        with self.assertRaises(ValueError):
            recent.batch(np.array([4]), world.embeddings)
        np.testing.assert_array_equal(recent.batch(np.array([5, 6]), world.embeddings)[2], [5, 6])

    def test_diagonal_detects_corruption_after_ring_wrap_and_target_updates(self):
        cfg = config()
        world = World(9, observation_dim=8)
        model, replay, anchor, counts = age(world, 11, cfg)
        context = world.post_context(anchor, "mixed")
        tape = generate(model, replay, world, context, cfg, 73, counts["optimizer_updates"])
        result = consume(model, replay, tape, world, context, cfg, diagonal=True)
        self.assertEqual(result["fingerprint"], tape.final_fingerprint)
        corrupted = tape.rows.copy()
        corrupted[:, 2] += 1.0
        with self.assertRaises(AssertionError):
            consume(model, replay, replace(tape, rows=corrupted), world, context, cfg, diagonal=True)

    def test_serialized_checkpoint_preserves_exact_learning_continuation(self):
        cfg = config()
        world = World(9, observation_dim=8)
        model, replay, context, counts = age(world, 11, cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_checkpoint(model, replay, path, "checkpoint.npz")
            restored, restored_replay = load_checkpoint(path / "checkpoint.npz", cfg, model.fingerprint())
            expected = generate(model, replay, world, context, cfg, 91, counts["optimizer_updates"])
            actual = generate(restored, restored_replay, world, context, cfg, 91, counts["optimizer_updates"])
            self.assertEqual(expected.final_fingerprint, actual.final_fingerprint)
            self.assertEqual(expected.digest(), actual.digest())
            self.assertEqual(expected.scores, actual.scores)

    def test_identity_intervention_has_no_causal_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_pair(9, 11, config(), Path(directory))["switches"]["mixed"]
        self.assertEqual(result["tape_digests"]["A"], result["tape_digests"]["I"])
        for name in ("total_effect", "learner_state_effect", "generated_data_effect", "interaction"):
            self.assertEqual(result[name], 0.0)


if __name__ == "__main__":
    unittest.main()
