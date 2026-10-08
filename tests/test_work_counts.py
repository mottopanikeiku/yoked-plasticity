"""Committed resource accounting must regenerate from the saved run records."""
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "summarize_work.py"
spec = importlib.util.spec_from_file_location("summarize_work", SCRIPT)
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)

REPORTS = {
    "work-a.json": "development-a",
    "work-b.json": "development-b",
    "work-a-v2.json": "development-a-v2",
    "work-b-v2.json": "development-b-v2",
    "work-additional-seeds.json": "additional-seeds",
}


class CommittedWorkCountTests(unittest.TestCase):
    def test_each_committed_report_matches_its_run(self):
        for report, run in REPORTS.items():
            with self.subTest(report=report):
                committed = json.loads((ROOT / "reports" / report).read_text())
                self.assertEqual(work.summarize(ROOT / "results" / run), committed)

    def test_repaired_runs_keep_every_original_count(self):
        for original, repaired in (("work-a.json", "work-a-v2.json"), ("work-b.json", "work-b-v2.json")):
            with self.subTest(report=original):
                before = json.loads((ROOT / "reports" / original).read_text())
                after = json.loads((ROOT / "reports" / repaired).read_text())
                self.assertEqual(before["counts"], after["counts"])

    def test_resource_table_in_development_report(self):
        # reports/development.md, "Actual resource use": (A, B) per quantity.
        quoted = {
            "training_simulator_interactions": (1_500_023, 1_500_026),
            "reused_consumer_transition_exposures": (810_000, 810_000),
            "neural_optimizer_updates": (434_454, 434_455),
            "tabular_minibatch_updates": (142_460, 142_460),
            "evaluation_simulator_interactions": (25_278, 24_096),
            "exact_reference_transition_queries": (648, 648),
            "instrumented_network_forward_examples": (48_604_693, 48_479_986),
            "derived_initial_equivalence_check_forward_examples": (240, 240),
        }
        counts = [json.loads((ROOT / "reports" / name).read_text())["counts"]
                  for name in ("work-a.json", "work-b.json")]
        for name, expected in quoted.items():
            with self.subTest(quantity=name):
                self.assertEqual((counts[0][name], counts[1][name]), expected)
        self.assertEqual(sum(c["training_simulator_interactions"] for c in counts), 3_000_049)


if __name__ == "__main__":
    unittest.main()
