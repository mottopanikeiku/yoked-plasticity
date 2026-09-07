"""Consumer-level configuration, clustered inference, and immutable-run checks."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from yoked_plasticity import cli


def configuration(**changes):
    config = {
        "observation_dim": 4, "hidden": 4, "learning_rate": 0.001,
        "gamma": 0.97, "replay_capacity": 16, "batch_size": 4, "epsilon": 0.1,
        "update_every": 1, "target_every": 4, "age_steps": 8, "block_steps": 4,
        "adapt_steps": 8, "evaluate_every": 4, "restart_steps": 4, "recent_keep": 8,
        "phase": "development", "structural_seeds": [7], "learner_seeds": [11, 12],
        "switches": ["visible_refit", "hidden_opportunity", "mixed"], "intervention": "injection",
    }
    config.update(changes)
    return config


def switch_result(g_a=-0.1, g_i=0.2, data_a=0.1):
    aa = 0.1
    ia, ai, ii = aa + g_a, aa + data_a, aa + data_a + g_i
    cells = {name: {"auc": value, "scores": [], "fingerprint": name,
                    "optimizer_updates": 1, "evaluation_interactions": 10}
             for name, value in {"AA": aa, "IA": ia, "AI": ai, "II": ii, "FA": 0.2, "FI": 0.2}.items()}
    return {
        "cells": cells, "total_effect": ii - aa,
        "learner_state_effect": ((ia - aa) + (ii - ai)) / 2,
        "generated_data_effect": ((ai - aa) + (ii - ia)) / 2,
        "interaction": ii - ia - ai + aa,
        "clean_tape_calibration": {name: {"auc": value} for name, value in {"A": 0.0, "I": 0.2, "O": 0.1}.items()},
        "controls": {},
    }


def pair_result(structural_seed, learner_seed, config=None, output=None):
    return {
        "structural_seed": structural_seed, "learner_seed": learner_seed,
        "age": {"training_interactions": 8, "optimizer_updates": 4},
        "elapsed_seconds": 0.125,
        "switches": {name: switch_result() for name in cli.SWITCHES},
    }


class ConfigurationTests(unittest.TestCase):
    def test_numeric_boundaries_reject_nonfinite_boolean_and_infeasible_inputs(self):
        cases = [
            ("observation_dim", 3), ("hidden", 0), ("batch_size", True),
            ("update_every", 1.5), ("target_every", -1), ("age_steps", 3),
            ("adapt_steps", 3), ("replay_capacity", 3), ("restart_steps", 0),
            ("learning_rate", 0), ("learning_rate", float("inf")),
            ("gamma", 0), ("gamma", 1.01), ("epsilon", -0.01),
            ("epsilon", 1.01), ("epsilon", float("nan")),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                cli.validate_config(configuration(**{name: value}))

    def test_exact_buffer_and_probability_boundaries_are_accepted(self):
        for epsilon in (0, 1):
            config = configuration(replay_capacity=4, age_steps=4, adapt_steps=4, gamma=1, epsilon=epsilon)
            self.assertEqual(cli.validate_config(config), config)

    def test_seed_and_protocol_boundaries(self):
        cases = [
            ("structural_seeds", []), ("learner_seeds", [1, 1]),
            ("structural_seeds", [-1]), ("learner_seeds", [False]),
            ("structural_seeds", [1.5]), ("phase", "production"),
            ("intervention", "optimizer_reset"), ("switches", []),
            ("switches", ["mixed", "mixed"]), ("switches", ["unknown"]),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                cli.validate_config(configuration(**{name: value}))
        self.assertEqual(cli.validate_config(configuration(structural_seeds=[0]))["structural_seeds"], [0])


class ClusterInferenceTests(unittest.TestCase):
    def test_structural_clusters_receive_equal_weight_not_init_count_weight(self):
        rows = [pair_result(7, seed) for seed in (11, 12, 13)] + [pair_result(8, 11)]
        for row in rows:
            row["switches"]["mixed"]["total_effect"] = 0.0 if row["structural_seed"] == 7 else 1.0
        estimate = cli.summarize(rows, configuration(switches=["mixed"]))["switches"]["mixed"]["total_effect"]
        self.assertEqual(estimate["mean"], 0.5)  # Pooled initialization mean would be 0.25.
        self.assertEqual(estimate["ci95"], [0.0, 1.0])
        self.assertEqual([row["value"] for row in estimate["structural_clusters"]], [0.0, 1.0])
        self.assertEqual([len(row["learner_seed_values"]) for row in estimate["structural_clusters"]], [3, 1])

    def test_single_cluster_does_not_bootstrap_initializations_as_independent_units(self):
        rows = [pair_result(7, 11), pair_result(7, 12)]
        for row, value in zip(rows, (-100.0, 100.0)):
            row["switches"]["mixed"]["total_effect"] = value
        estimate = cli.summarize(rows, configuration(switches=["mixed"]))["switches"]["mixed"]["total_effect"]
        self.assertEqual(estimate["mean"], 0.0)
        self.assertEqual(estimate["ci95"], [0.0, 0.0])

    def test_semantic_digest_ignores_only_elapsed_seconds_recursively(self):
        original = pair_result(7, 11)
        original["switches"]["mixed"]["elapsed_seconds"] = 1.0
        changed = copy.deepcopy(original)
        changed["elapsed_seconds"] = 100.0
        changed["switches"]["mixed"]["elapsed_seconds"] = 200.0
        self.assertEqual(cli.semantic_digest(original), cli.semantic_digest(changed))
        changed["switches"]["mixed"]["cells"]["IA"]["auc"] += 0.001
        self.assertNotEqual(cli.semantic_digest(original), cli.semantic_digest(changed))

    def test_all_three_scientific_gates_are_required_independently(self):
        baseline = pair_result(7, 11)
        summary = cli.summarize([baseline], configuration())
        self.assertEqual(summary["scientific_gate"], "advance")
        estimate = summary["switches"]["mixed"]["conditional_learner_gain_A"]
        self.assertAlmostEqual(estimate["mean"], -0.1)
        self.assertEqual(estimate["ci95"], [estimate["mean"], estimate["mean"]])
        for failing_gate in ("clean", "hidden", "reversal"):
            row = copy.deepcopy(baseline)
            if failing_gate == "clean":
                for switch in ("visible_refit", "mixed"):
                    row["switches"][switch]["clean_tape_calibration"]["I"]["auc"] = 0.05
            elif failing_gate == "hidden":
                row["switches"]["hidden_opportunity"] = switch_result(data_a=-0.15)
            else:
                row["switches"] = {switch: switch_result(g_a=0.1, g_i=0.1) for switch in cli.SWITCHES}
            with self.subTest(failing_gate=failing_gate):
                stopped = cli.summarize([row], configuration())
                self.assertEqual(stopped["scientific_gate"], "stop")
                self.assertTrue(stopped["numerical_run_success"])

    def test_reversal_is_on_cluster_means_not_any_individual_seed(self):
        rows = [pair_result(7, 11), pair_result(7, 12)]
        for row, g_a in zip(rows, (-0.2, 0.4)):
            row["switches"] = {switch: switch_result(g_a=g_a, g_i=0.1) for switch in cli.SWITCHES}
        summary = cli.summarize(rows, configuration())
        self.assertFalse(summary["gate_details"]["conditional_reversal_pass"])
        self.assertEqual(summary["scientific_gate"], "stop")

    def test_smoke_and_nonprimary_interventions_never_qualify(self):
        for changes in ({"phase": "smoke"}, {"intervention": "identity"}, {"intervention": "head_reset"}):
            with self.subTest(changes=changes):
                summary = cli.summarize([pair_result(7, 11)], configuration(**changes))
                self.assertEqual(summary["scientific_gate"], "stop")
                self.assertTrue(summary["numerical_run_success"])


class RunLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "input.json"
        self.config_path.write_text(json.dumps(configuration(), indent=3) + "\n", encoding="utf-8")

    def execute(self, output, function=pair_result, prerequisite=None):
        # Exercise real filesystem lifecycle and inference without any learning.
        with patch.dict("sys.modules", {"yoked_plasticity.experiment": SimpleNamespace(run_pair=function)}):
            with redirect_stdout(io.StringIO()):
                return cli.run(self.config_path, output, prerequisite)

    def test_existing_directory_is_untouched_even_when_empty(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.execute(output)
        self.assertEqual(list(output.iterdir()), [])
        sentinel = output / "manifest.json"
        sentinel.write_bytes(b"original run")
        with self.assertRaises(FileExistsError):
            self.execute(output)
        self.assertEqual(sentinel.read_bytes(), b"original run")
        self.assertFalse((output / "failure.json").exists())

    def test_pair_failure_retains_completed_raw_values_and_marks_failure(self):
        output = self.root / "failed"

        def fail_second(structural_seed, learner_seed, config, directory):
            if learner_seed == 12:
                raise RuntimeError("intentional learning failure")
            return pair_result(structural_seed, learner_seed)

        with self.assertRaises(RuntimeError):
            self.execute(output, fail_second)
        rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual([(row["structural_seed"], row["learner_seed"]) for row in rows], [(7, 11)])
        failure = json.loads((output / "failure.json").read_text())
        self.assertEqual(failure["completed_pairs"], 1)
        self.assertIn("RuntimeError: intentional learning failure", failure["traceback"])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["numerical_run_success"])
        self.assertFalse((output / "summary.json").exists())

    def test_invalid_configuration_is_an_explicit_failed_run(self):
        self.config_path.write_text(json.dumps(configuration(batch_size=0)), encoding="utf-8")
        output = self.root / "invalid"
        with self.assertRaises(ValueError):
            self.execute(output)
        self.assertEqual((output / "config.json").read_bytes(), self.config_path.read_bytes())
        self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "failed")
        self.assertFalse((output / "results.jsonl").exists())

    def test_heldout_requires_hash_bound_success_and_disjoint_core_matched_config(self):
        development = self.root / "development"
        self.execute(development)
        self.assertEqual((development / "config.json").read_bytes(), self.config_path.read_bytes())
        heldout = configuration(phase="heldout", structural_seeds=[8], learner_seeds=[13])
        binding = cli.validate_prerequisite(development, heldout)
        self.assertEqual(binding["directory"], str(development.resolve()))
        for changed in (
            {"structural_seeds": [7]}, {"learner_seeds": [12]},
            {"learning_rate": 0.002}, {"intervention": "identity"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                cli.validate_prerequisite(development, {**heldout, **changed})
        with (development / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaises(ValueError):
            cli.validate_prerequisite(development, heldout)

    def test_numerically_successful_stop_cannot_authorize_heldout(self):
        def no_reversal(structural_seed, learner_seed, config, directory):
            result = pair_result(structural_seed, learner_seed)
            result["switches"] = {switch: switch_result(g_a=0.1, g_i=0.1) for switch in cli.SWITCHES}
            return result

        development = self.root / "stopped"
        summary = self.execute(development, no_reversal)
        self.assertTrue(summary["numerical_run_success"])
        self.assertEqual(summary["scientific_gate"], "stop")
        with self.assertRaises(ValueError):
            cli.validate_prerequisite(development, configuration(phase="heldout", structural_seeds=[8], learner_seeds=[13]))

    def test_heldout_without_prerequisite_fails_before_any_pairs(self):
        self.config_path.write_text(json.dumps(configuration(phase="heldout")), encoding="utf-8")
        output = self.root / "heldout"
        with self.assertRaises(ValueError):
            self.execute(output)
        self.assertFalse((output / "results.jsonl").exists())
        self.assertFalse(json.loads((output / "manifest.json").read_text())["numerical_run_success"])


if __name__ == "__main__":
    unittest.main()
