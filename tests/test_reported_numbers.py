"""Numbers quoted in the README and reports must match the committed results."""
import json
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
SWITCHES = {"Visible": "visible_refit", "Hidden": "hidden_opportunity", "Mixed": "mixed"}


def load(relative):
    return json.loads((ROOT / relative).read_text())


def records(run):
    return [json.loads(line) for line in (ROOT / "results" / run / "results.jsonl").read_text().splitlines()]


def table_rows(relative, first_cells):
    """Return numeric cells of markdown rows whose leading cells match."""
    rows = []
    for line in (ROOT / relative).read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and tuple(cells[:len(first_cells)]) == first_cells:
            rows.append(cells)
    return rows


def number(cell):
    return float(cell.split()[0].replace(",", ""))


class DevelopmentReportTests(unittest.TestCase):
    def test_primary_effects_table(self):
        metrics = ("total_effect", "learner_state_effect", "generated_data_effect",
                   "conditional_learner_gain_A", "conditional_learner_gain_I", "clean_gain", "excess_clean_gain")
        for label, run in (("A", "development-a-v2"), ("B", "development-b-v2")):
            summary = load(f"results/{run}/summary.json")
            for name, switch in SWITCHES.items():
                with self.subTest(run=label, switch=name):
                    (row,) = [r for r in table_rows("reports/development.md", (label, name)) if len(r) == 9]
                    expected = [round(summary["switches"][switch][metric]["mean"], 5) for metric in metrics]
                    self.assertEqual([number(cell) for cell in row[2:]], expected)

    def test_baseline_auc_table(self):
        for label, run in (("A", "development-a-v2"), ("B", "development-b-v2")):
            pairs = records(run)
            for name, switch in SWITCHES.items():
                with self.subTest(run=label, switch=name):
                    (row,) = [r for r in table_rows("reports/development.md", (label, name)) if len(r) == 7]
                    fields = [lambda s: s["cells"]["AA"]["auc"], lambda s: s["cells"]["II"]["auc"],
                              lambda s: s["controls"]["epsilon_restart"]["auc"],
                              lambda s: s["controls"]["recent_replay"]["auc"],
                              lambda s: s["controls"]["tabular_recent_replay"]["auc"]]
                    expected = [round(statistics.fmean(f(p["switches"][switch]) for p in pairs), 5) for f in fields]
                    self.assertEqual([number(cell) for cell in row[2:]], expected)

    def test_both_configurations_stop(self):
        for run in ("development-a-v2", "development-b-v2", "additional-seeds"):
            self.assertEqual(load(f"results/{run}/summary.json")["scientific_gate"], "stop")


class AdditionalSeedReportTests(unittest.TestCase):
    def test_aggregate_table(self):
        summary = load("results/additional-seeds/summary.json")
        pairs = records("additional-seeds")
        for name, switch in SWITCHES.items():
            with self.subTest(switch=name):
                (row,) = table_rows("reports/additional-seeds.md", (name,))
                estimates = summary["switches"][switch]
                for cell, metric in ((row[1], "conditional_learner_gain_A"), (row[2], "conditional_learner_gain_I")):
                    mean, interval = cell.split(" ", 1)
                    low, high = (float(value) for value in interval.strip("[]").split(","))
                    self.assertEqual(float(mean), round(estimates[metric]["mean"], 5))
                    self.assertEqual([low, high], [round(value, 5) for value in estimates[metric]["ci95"]])
                self.assertEqual(number(row[3]), round(estimates["total_effect"]["mean"], 5))
                self.assertEqual(number(row[4]), round(estimates["generated_data_effect"]["mean"], 5))
                tabular = statistics.fmean(p["switches"][switch]["controls"]["tabular_recent_replay"]["auc"] for p in pairs)
                self.assertEqual(number(row[5]), round(tabular, 5))


class ReadmeTests(unittest.TestCase):
    def test_saved_tape_gain_table(self):
        comparison = load("results/tape-comparison.json")
        for case in comparison["cases"]:
            short = {"visible_refit": "visible", "mixed": "mixed"}[case["switch"]]
            for entry in case["initializations"]:
                with self.subTest(world=case["structural_seed"], learner=entry["learner_seed"]):
                    (row,) = table_rows("README.md", (f"{case['structural_seed']} / {short}", str(entry["learner_seed"])))
                    self.assertEqual([number(row[2]), number(row[3])],
                                     [round(entry["g_A"], 5), round(entry["g_I"], 5)])

    def test_additional_seed_numbers_in_prose(self):
        text = (ROOT / "README.md").read_text()
        summary = load("results/additional-seeds/summary.json")["switches"]["hidden_opportunity"]
        pairs = records("additional-seeds")
        tabular = [statistics.fmean(p["switches"][s]["controls"]["tabular_recent_replay"]["auc"] for p in pairs)
                   for s in ("hidden_opportunity", "mixed")]
        self.assertIn(f"hidden tape-conditioned gains of {summary['conditional_learner_gain_A']['mean']:.5f}"
                      f"/{summary['conditional_learner_gain_I']['mean']:.5f}", text)
        self.assertIn(f"tabular hidden/mixed AUCs of {tabular[0]:.5f}/{tabular[1]:.5f}", text)


if __name__ == "__main__":
    unittest.main()
