"""Branch counts and sampled adaptation timing have distinct meanings."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from yoked_plasticity.env import Context, World


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_saved_tapes.py"
spec = importlib.util.spec_from_file_location("saved_tapes", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class SavedTapeTests(unittest.TestCase):
    def test_branch_entries_depths_successes_and_one_based_steps(self):
        rows = np.array([[0, 1, 0, 5, 0], [5, 0, 0, 6, 0], [6, 0, 0, 7, 0],
                         [7, 0, 0, 8, 0], [8, 0, 1.5, 9, 1],
                         [0, 0, 0, 1, 0], [1, 1, -0.05, 9, 1]], dtype=float)
        zero, one = analysis.coverage(rows, offset=1000)
        self.assertEqual(zero, {"branch": 0, "entries": 1, "depth_visits": [1, 0, 0, 0],
                                "successes": 0, "first_success_step": None})
        self.assertEqual(one, {"branch": 1, "entries": 1, "depth_visits": [1, 1, 1, 1],
                               "successes": 1, "first_success_step": 1005})

    def test_empty_blocks_have_no_invented_success(self):
        self.assertEqual(analysis.coverage(np.empty((0, 5)))[1]["successes"], 0)
        self.assertIsNone(analysis.coverage(np.empty((0, 5)))[0]["first_success_step"])

    def test_first_optimal_is_not_sustained_optimal(self):
        scores = [{"step": index * 250, "normalized_return": value}
                  for index, value in enumerate((0, 1, 0, 1, 1))]
        measured = analysis.timing(scores)
        self.assertEqual(measured["first_optimal_step"], 250)
        self.assertEqual(measured["optimal_through_last_sample_from_step"], 750)
        self.assertEqual(measured["optimal_sample_count"], 3)

    def test_no_optimum_is_missing_not_horizon(self):
        measured = analysis.timing([{"step": 0, "normalized_return": 0},
                                    {"step": 5000, "normalized_return": 0.2}])
        self.assertIsNone(measured["first_optimal_step"])
        self.assertIsNone(measured["optimal_through_last_sample_from_step"])

    def test_final_sample_optimum_is_only_final_sample(self):
        measured = analysis.timing([{"step": 0, "normalized_return": 0},
                                    {"step": 5000, "normalized_return": 1}])
        self.assertEqual(measured["optimal_through_last_sample_from_step"], 5000)

    def test_chronology_and_reward_checked_against_saved_context(self):
        world = World(404)
        context = Context(codes=((0, 0, 0, 0), (1, 1, 1, 1)), rewards=(0.2, 1.5))
        rows = np.array([[0, 0, 0, 1, 0], [1, 1, -0.05, 9, 1]], dtype=float)
        analysis.validate_tape(rows, world, context)
        for column, wrong in ((0, 2), (2, 0.2), (3, 4), (4, 0)):
            changed = rows.copy()
            changed[1, column] = wrong
            with self.assertRaises(ValueError):
                analysis.validate_tape(changed, world, context)

    def test_committed_result_matches_saved_sources(self):
        root = SCRIPT.parents[1]
        rebuilt = analysis.analyze(root / "results" / "additional-seeds")
        import json
        committed = json.loads((root / "results" / "tape-comparison.json").read_text())
        # JSON normalizes context tuples into lists.
        self.assertEqual(json.loads(json.dumps(rebuilt)), committed)
        self.assertEqual(analysis.report(rebuilt), (root / "reports" / "tape-comparison.md").read_text())
        self.assertEqual(rebuilt["original_aggregate_decision"], "stop")
        self.assertEqual([len(case["initializations"]) for case in rebuilt["cases"]], [2, 2])


if __name__ == "__main__":
    unittest.main()
