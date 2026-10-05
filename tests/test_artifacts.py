"""Behavioral audits of generated tiny artifacts, without learning imports."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


VERIFIER = Path(__file__).resolve().parents[1] / "tools" / "verify_artifacts.py"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path, value):
    path.write_bytes(canonical(value))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    write_json(root / "output-hashes.json", {
        "algorithm": "sha256", "excludes": ["output-hashes.json"],
        "files": {path.relative_to(root).as_posix(): digest(path)
                  for path in root.rglob("*") if path.is_file() and path.name != "output-hashes.json"},
    })


def records(root):
    return [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def write_records(root, rows):
    (root / "results.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def bind_semantics(root):
    def omit_timing(value):
        if isinstance(value, dict):
            return {key: omit_timing(item) for key, item in value.items() if key != "elapsed_seconds"}
        if isinstance(value, list):
            return [omit_timing(item) for item in value]
        return value

    semantic = hashlib.sha256(canonical(omit_timing(records(root)))).hexdigest()
    for filename in ("manifest.json", "summary.json"):
        record = load_json(root / filename)
        record["semantic_sha256"] = semantic
        write_json(root / filename, record)
    inventory(root)


def fixture(root, timing=1.0, aa=0.25):
    root.mkdir()
    write_json(root / "config.json", {
        "phase": "development", "intervention": "injection",
        "structural_seeds": [7], "learner_seeds": [11, 12], "switches": ["mixed"],
    })
    # The audit binds historic source hashes, never opens these paths.
    sources = {"historic/absent.py": hashlib.sha256(b"historic learning source").hexdigest()}
    write_json(root / "manifest.json", {
        "status": "success", "numerical_run_success": True, "scientific_gate": "stop",
        "phase": "development", "intervention": "injection", "completed_pairs": 2,
        "config_sha256": digest(root / "config.json"), "source_sha256": sources,
        "source_binding_sha256": hashlib.sha256(canonical(sources)).hexdigest(),
        "walltime_seconds": timing, "command": ["run", str(root)],
    })
    expected = {
        "total_effect": 0.75 - aa,
        "learner_state_effect": (-aa + 0.25) / 2,
        "generated_data_effect": ((0.5 - aa) + 0.75) / 2,
        "interaction": 0.25 + aa,
        "clean_gain": 0.5, "excess_clean_gain": 0.25,
        "conditional_learner_gain_A": -aa, "conditional_learner_gain_I": 0.25,
    }
    cells = {name: {"auc": value, "diagonal_verified": name in ("AA", "II"),
                    "scores": [{"step": 0, "normalized_return": value, "elapsed_seconds": timing}]}
             for name, value in {"AA": aa, "IA": 0.0, "AI": 0.5, "II": 0.75, "FA": 0.0, "FI": 0.0}.items()}
    fields = {
        "cells": cells,
        "clean_tape_calibration": {name: {"auc": value, "diagonal_verified": name == "A"}
                                   for name, value in {"A": 0.0, "I": 0.5, "O": 0.25}.items()},
        **{key: expected[key] for key in ("total_effect", "learner_state_effect", "generated_data_effect", "interaction")},
        "elapsed_seconds": timing,
    }
    write_records(root, [{"structural_seed": 7, "learner_seed": learner, "switches": {"mixed": fields},
                          "elapsed_seconds": timing} for learner in (11, 12)])
    estimates = {
        metric: {"mean": value, "ci95": [value, value], "structural_cluster_count": 1,
                 "structural_clusters": [{"structural_seed": 7, "value": value,
                                          "learner_seed_values": [{"learner_seed": learner, "value": value}
                                                                  for learner in (11, 12)]}]}
        for metric, value in expected.items()
    }
    write_json(root / "summary.json", {
        "status": "success", "numerical_run_success": True, "scientific_gate": "stop",
        "phase": "development", "intervention": "injection", "pair_count": 2,
        "switches": {"mixed": estimates}, "walltime_seconds": timing,
    })
    for learner in (11, 12):
        # Opaque stand-ins: this verifier hashes NPZ bytes, it does not decode them.
        for filename in (f"checkpoint-7-{learner}.npz", f"tapes-7-{learner}-mixed.npz"):
            (root / filename).write_bytes(b"tiny raw artifact fixture\x00" + filename.encode("ascii"))
    bind_semantics(root)
    return root


class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = fixture(self.base / "run")

    def audit(self, *args, success=True):
        result = subprocess.run([sys.executable, "-B", str(VERIFIER), str(self.root), *map(str, args)],
                                capture_output=True, text=True, check=False, cwd=self.base)
        verdict = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0 if success else 1, result.stderr + result.stdout)
        self.assertIs(verdict["ok"], success)
        return verdict

    def test_successful_negative_scientific_result_is_valid_and_read_only(self):
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        verdict = self.audit()
        self.assertEqual(verdict["scientific_gate"], "stop")
        self.assertEqual(verdict["verified_artifact_count"], 8)
        self.assertEqual({path.name: path.read_bytes() for path in self.root.iterdir()}, before)

    def test_raw_binary_hash_tampering_fails(self):
        (self.root / "tapes-7-11-mixed.npz").write_bytes(b"corrupted raw tape")
        self.audit(success=False)

    def test_raw_record_tampering_fails_even_when_inventory_is_refreshed(self):
        rows = records(self.root)
        rows[0]["switches"]["mixed"]["cells"]["FA"]["auc"] += 0.125
        write_records(self.root, rows)
        inventory(self.root)
        self.audit(success=False)

    def test_algebra_tampering_fails_even_when_all_digests_are_refreshed(self):
        rows = records(self.root)
        rows[0]["switches"]["mixed"]["generated_data_effect"] += 0.125
        write_records(self.root, rows)
        bind_semantics(self.root)
        self.audit(success=False)

    def test_summary_conditional_gain_must_match_raw_cells(self):
        summary = load_json(self.root / "summary.json")
        estimate = summary["switches"]["mixed"]["conditional_learner_gain_A"]
        estimate["mean"] = 0.0
        write_json(self.root / "summary.json", summary)
        inventory(self.root)
        self.audit(success=False)

    def test_missing_artifact_is_rejected_even_if_removed_from_inventory(self):
        (self.root / "tapes-7-11-mixed.npz").unlink()
        self.audit(success=False)
        inventory(self.root)
        self.audit(success=False)

    def test_unlisted_artifact_is_rejected(self):
        (self.root / "unlisted.bin").write_bytes(b"unexpected")
        self.audit(success=False)

    def test_inventory_paths_cannot_escape_root(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"outside artifact")
        original = load_json(self.root / "output-hashes.json")
        for relative in ("../outside.bin", str(outside)):
            changed = copy.deepcopy(original)
            changed["files"][relative] = digest(outside)
            write_json(self.root / "output-hashes.json", changed)
            with self.subTest(relative=relative):
                self.audit(success=False)

    def test_symlink_escape_is_rejected(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"outside artifact")
        path = self.root / "tapes-7-11-mixed.npz"
        path.unlink()
        path.symlink_to(outside)
        inventory(self.root)
        self.audit(success=False)

    def test_duplicate_pair_cannot_replace_missing_configured_pair(self):
        rows = records(self.root)
        rows[1] = copy.deepcopy(rows[0])
        write_records(self.root, rows)
        bind_semantics(self.root)
        self.audit(success=False)

    def test_all_diagonal_flags_are_required(self):
        original = records(self.root)
        for cell in ("AA", "II", "clean A"):
            rows = copy.deepcopy(original)
            fields = rows[0]["switches"]["mixed"]
            target = fields["clean_tape_calibration"]["A"] if cell == "clean A" else fields["cells"][cell]
            target["diagonal_verified"] = False
            write_records(self.root, rows)
            bind_semantics(self.root)
            with self.subTest(cell=cell):
                self.audit(success=False)

    def test_numerical_status_and_phase_must_agree(self):
        original = load_json(self.root / "summary.json")
        for field, value in (("numerical_run_success", False),
                             ("phase", "heldout"),
                             ("intervention", "identity")):
            summary = {**original, field: value}
            write_json(self.root / "summary.json", summary)
            inventory(self.root)
            with self.subTest(field=field):
                self.audit(success=False)

    def test_configuration_bytes_and_source_mapping_are_independently_bound(self):
        original = (self.root / "config.json").read_bytes()
        (self.root / "config.json").write_bytes(original + b"\n")
        inventory(self.root)
        self.audit(success=False)
        (self.root / "config.json").write_bytes(original)
        manifest = load_json(self.root / "manifest.json")
        manifest["source_sha256"]["historic/absent.py"] = "0" * 64
        write_json(self.root / "manifest.json", manifest)
        inventory(self.root)
        self.audit(success=False)

    def test_comparison_ignores_recursive_timing_and_directory_names(self):
        other = fixture(self.base / "different-name", timing=99.0)
        verdict = self.audit("--compare", other)
        self.assertIs(verdict["comparison_equal"], True)
        self.assertEqual(verdict["verified_artifact_count"], 16)

    def test_comparison_rejects_different_scientific_values(self):
        other = fixture(self.base / "different-science", aa=0.125)
        self.audit("--compare", other, success=False)

    def test_comparison_rejects_different_historic_sources(self):
        other = fixture(self.base / "different-source")
        manifest = load_json(other / "manifest.json")
        manifest["source_sha256"]["historic/absent.py"] = "0" * 64
        manifest["source_binding_sha256"] = hashlib.sha256(canonical(manifest["source_sha256"])).hexdigest()
        write_json(other / "manifest.json", manifest)
        inventory(other)
        self.audit("--compare", other, success=False)

    def test_comparison_verifies_the_other_run_before_comparing(self):
        other = fixture(self.base / "damaged-comparison")
        (other / "checkpoint-7-11.npz").unlink()
        self.audit("--compare", other, success=False)

    def test_comparison_rejects_different_bound_configurations(self):
        other = fixture(self.base / "different-config")
        config = load_json(other / "config.json")
        config["learning_rate"] = 0.001
        write_json(other / "config.json", config)
        manifest = load_json(other / "manifest.json")
        manifest["config_sha256"] = digest(other / "config.json")
        write_json(other / "manifest.json", manifest)
        inventory(other)
        self.audit("--compare", other, success=False)


if __name__ == "__main__":
    unittest.main()
