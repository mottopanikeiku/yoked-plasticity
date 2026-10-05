"""Audit immutable run consistency using only the Python standard library.

This is a consistency check, not a trusted adversarial signature: somebody who
rewrites all artifacts and their hashes can forge a consistent run. Historical
source mappings are bound without reading current source files. NPZ files are
hash-checked, not replayed; recorded diagonal flags are checked, not re-executed.
Bootstrap intervals, learning correctness, and scientific confirmation are not
established by this audit. Run directories must be quiescent and symlink-free.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys


METRICS = (
    "total_effect", "learner_state_effect", "generated_data_effect", "interaction",
    "clean_gain", "excess_clean_gain", "conditional_learner_gain_A",
    "conditional_learner_gain_I",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def without_elapsed(value):
    if isinstance(value, dict):
        return {key: without_elapsed(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_elapsed(item) for item in value]
    return value


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    # Also rejects overflowing numeric literals such as 1e999.
    canonical(value)
    return value


def load_json(path):
    return parse_json(path.read_text(encoding="utf-8"))


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def artifact_path(root, relative):
    require(isinstance(relative, str) and relative, "artifact path must be a nonempty string")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and path.as_posix() == relative
            and ".." not in path.parts and relative != "." and "\\" not in relative,
            f"unsafe artifact path: {relative}")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root), f"artifact path escapes run root: {relative}")
    return resolved


def verify_inventory(root):
    require(root.is_dir(), f"not a run directory: {root}")
    inventory_path = root / "output-hashes.json"
    require(not inventory_path.is_symlink(), "symlink artifact: output-hashes.json")
    inventory = load_json(inventory_path)
    require(inventory.get("algorithm") == "sha256", "inventory algorithm must be sha256")
    require(inventory.get("excludes") == ["output-hashes.json"], "unexpected inventory exclusions")
    hashes = inventory["files"]
    require(isinstance(hashes, dict), "inventory files must be an object")
    required = {"config.json", "manifest.json", "summary.json", "results.jsonl"}
    require(required <= hashes.keys(), "inventory missing required artifacts")
    require("output-hashes.json" not in hashes, "inventory cannot hash itself")
    for relative, expected in hashes.items():
        artifact_path(root, relative)
        require(valid_digest(expected), f"invalid SHA256 for {relative}")
    actual = set()

    def walk_error(error):
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in directories + files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            require(not path.is_symlink(), f"symlink artifact: {relative}")
            require(path.is_dir() or path.is_file(), f"nonregular artifact: {relative}")
            if path.is_file() and relative != "output-hashes.json":
                actual.add(relative)
    require(actual == set(hashes),
            f"artifact inventory mismatch: missing={sorted(set(hashes) - actual)}, "
            f"unlisted={sorted(actual - set(hashes))}")
    require("failure.json" not in actual, "successful run contains failure.json")
    for relative, expected in hashes.items():
        require(sha256_file(artifact_path(root, relative)) == expected, f"SHA256 mismatch: {relative}")
    return hashes


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric")
    return value


def close(actual, expected, label):
    require(math.isclose(number(actual, label), number(expected, label), rel_tol=1e-12, abs_tol=1e-12),
            f"algebra mismatch: {label}: {actual!r} != {expected!r}")


def seed_list(config, key):
    values = config[key]
    require(isinstance(values, list) and values, f"{key} must be a nonempty list")
    require(all(type(value) is int and value >= 0 for value in values), f"invalid {key}")
    require(len(set(values)) == len(values), f"duplicate {key}")
    return values


def pair_metrics(fields, label):
    cells = fields["cells"]
    require(set(cells) == {"AA", "IA", "AI", "II", "FA", "FI"}, f"{label}: incorrect primary cells")
    for name in ("AA", "II"):
        require(cells[name].get("diagonal_verified") is True, f"{label}/{name}: diagonal not verified")
    values = {name: number(cell["auc"], f"{label}/{name}/auc") for name, cell in cells.items()}
    aa, ia, ai, ii = (values[name] for name in ("AA", "IA", "AI", "II"))
    clean = fields["clean_tape_calibration"]
    require(set(clean) == {"A", "I", "O"}, f"{label}: incorrect clean calibration cells")
    require(clean["A"].get("diagonal_verified") is True, f"{label}/clean A: diagonal not verified")
    clean_values = {name: number(cell["auc"], f"{label}/clean/{name}/auc") for name, cell in clean.items()}
    metrics = {
        "total_effect": ii - aa,
        "learner_state_effect": ((ia - aa) + (ii - ai)) / 2,
        "generated_data_effect": ((ai - aa) + (ii - ia)) / 2,
        "interaction": ii - ia - ai + aa,
        "clean_gain": clean_values["I"] - clean_values["A"],
        "excess_clean_gain": clean_values["I"] - clean_values["O"],
        "conditional_learner_gain_A": ia - aa,
        "conditional_learner_gain_I": ii - ai,
    }
    for metric in METRICS[:4]:
        close(fields[metric], metrics[metric], f"{label}/{metric}")
    for metric in METRICS[4:]:
        if metric in fields:
            close(fields[metric], metrics[metric], f"{label}/{metric}")
    close(fields["total_effect"], fields["learner_state_effect"] + fields["generated_data_effect"],
          f"{label}/allocation sum")
    close(fields["interaction"], metrics["conditional_learner_gain_I"] - metrics["conditional_learner_gain_A"],
          f"{label}/conditional interaction")
    return metrics


def verify_summary(summary, metrics, structures, learners, switches):
    require(set(summary["switches"]) == set(switches), "summary switch set mismatch")
    means = {}
    for switch in switches:
        means[switch] = {}
        for metric in METRICS:
            estimate = summary["switches"][switch][metric]
            clusters = estimate["structural_clusters"]
            require(len(clusters) == len(structures), f"{switch}/{metric}: cluster count mismatch")
            require(type(estimate["structural_cluster_count"]) is int
                    and estimate["structural_cluster_count"] == len(structures), "reported cluster count mismatch")
            seen = set()
            expected_clusters = []
            for cluster in clusters:
                seed = cluster["structural_seed"]
                require(type(seed) is int and seed in structures and seed not in seen, "invalid summary structural seed")
                seen.add(seed)
                raw = cluster["learner_seed_values"]
                require(len(raw) == len(learners), "summary learner count mismatch")
                seen_learners = set()
                for row in raw:
                    learner = row["learner_seed"]
                    require(type(learner) is int and learner in learners and learner not in seen_learners,
                            "invalid summary learner seed")
                    seen_learners.add(learner)
                    close(row["value"], metrics[(seed, learner)][switch][metric], f"{switch}/{metric}/learner")
                expected = math.fsum(metrics[(seed, learner)][switch][metric] for learner in sorted(learners)) / len(learners)
                close(cluster["value"], expected, f"{switch}/{metric}/cluster")
                expected_clusters.append(expected)
            expected = math.fsum(expected_clusters) / len(structures)
            close(estimate["mean"], expected, f"{switch}/{metric}/mean")
            means[switch][metric] = expected
    return means


def verify_run(directory):
    root = Path(directory).resolve()
    hashes = verify_inventory(root)
    config = load_json(root / "config.json")
    manifest = load_json(root / "manifest.json")
    summary = load_json(root / "summary.json")
    require(config["phase"] in ("smoke", "development", "heldout"), "invalid phase")
    require(config["intervention"] in ("injection", "head_reset", "identity"), "invalid intervention")
    for name, record in (("manifest", manifest), ("summary", summary)):
        require(record.get("status") == "success" and record.get("numerical_run_success") is True,
                f"{name}: numerical run not successful")
        for field in ("phase", "intervention"):
            require(record.get(field) == config[field], f"{name}: {field} mismatch")
        require(record.get("scientific_gate") in ("advance", "stop"), f"{name}: invalid scientific gate")
    require(manifest["scientific_gate"] == summary["scientific_gate"], "scientific gates disagree")
    config_digest = sha256_file(root / "config.json")
    require(manifest.get("config_sha256") == config_digest, "config SHA256 binding mismatch")
    if "config_sha256" in summary:
        require(summary["config_sha256"] == config_digest, "summary config SHA256 binding mismatch")
    source_hashes = manifest["source_sha256"]
    require(isinstance(source_hashes, dict) and source_hashes, "source mapping must be nonempty")
    require(all(isinstance(name, str) and name and valid_digest(value) for name, value in source_hashes.items()),
            "invalid source SHA256 mapping")
    source_digest = hashlib.sha256(canonical(source_hashes)).hexdigest()
    require(manifest.get("source_binding_sha256") == source_digest, "source binding SHA256 mismatch")
    with (root / "results.jsonl").open(encoding="utf-8") as handle:
        results = [parse_json(line) for line in handle]
    semantic_digest = hashlib.sha256(canonical(without_elapsed(results))).hexdigest()
    require(manifest.get("semantic_sha256") == semantic_digest
            and summary.get("semantic_sha256") == semantic_digest, "semantic SHA256 mismatch")
    structures = seed_list(config, "structural_seeds")
    learners = seed_list(config, "learner_seeds")
    switches = config["switches"]
    require(isinstance(switches, list) and switches
            and all(switch in ("visible_refit", "hidden_opportunity", "mixed") for switch in switches)
            and len(set(switches)) == len(switches), "invalid configured switches")
    expected_pairs = {(structure, learner) for structure in structures for learner in learners}
    require(len(results) == len(expected_pairs), "result pair count mismatch")
    for record, key in ((summary, "pair_count"), (manifest, "completed_pairs")):
        require(type(record[key]) is int and record[key] == len(expected_pairs), f"{key} mismatch")
    metrics = {}
    for result in results:
        pair = (result["structural_seed"], result["learner_seed"])
        require(all(type(seed) is int for seed in pair) and pair in expected_pairs and pair not in metrics,
                f"unexpected or duplicate result pair: {pair}")
        for field in ("phase", "intervention"):
            if field in result:
                require(result[field] == config[field], f"result {field} mismatch")
        require(set(result["switches"]) == set(switches), f"{pair}: result switch set mismatch")
        structure, learner = pair
        require(f"checkpoint-{structure}-{learner}.npz" in hashes, f"{pair}: missing raw checkpoint")
        metrics[pair] = {}
        for switch in switches:
            require(f"tapes-{structure}-{learner}-{switch}.npz" in hashes, f"{pair}/{switch}: missing raw tapes")
            metrics[pair][switch] = pair_metrics(result["switches"][switch], f"{pair}/{switch}")
    means = verify_summary(summary, metrics, structures, learners, switches)
    clean_pass = any(means[switch]["clean_gain"] >= 0.10 and means[switch]["excess_clean_gain"] >= 0.05
                     for switch in ("visible_refit", "mixed") if switch in means)
    hidden = means.get("hidden_opportunity")
    hidden_pass = hidden is not None and hidden["total_effect"] >= 0.05 and hidden["generated_data_effect"] >= 0.05 and hidden["generated_data_effect"] >= hidden["total_effect"] / 3
    reversal_pass = any(values["conditional_learner_gain_A"] * values["conditional_learner_gain_I"] < 0
                        and min(abs(values["conditional_learner_gain_A"]), abs(values["conditional_learner_gain_I"])) >= 0.05
                        and abs(values["conditional_learner_gain_I"] - values["conditional_learner_gain_A"]) >= 0.15
                        for values in means.values())
    eligible = config["intervention"] == "injection" and config["phase"] != "smoke"
    gate = "advance" if eligible and clean_pass and hidden_pass and reversal_pass else "stop"
    require(summary["scientific_gate"] == gate, "scientific gate inconsistent with primary means")
    return {"ok": True, "run_dir": str(root), "scientific_gate": gate,
            "verified_artifact_count": len(hashes), "config_sha256": config_digest,
            "source_binding_sha256": source_digest, "semantic_sha256": semantic_digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="completed immutable run directory")
    parser.add_argument("--compare", metavar="OTHER_RUN_DIR", help="verify a second run and require semantic equality")
    args = parser.parse_args(argv)
    try:
        verdict = verify_run(args.run_dir)
        if args.compare is not None:
            other = verify_run(args.compare)
            for field in ("config_sha256", "source_binding_sha256", "semantic_sha256"):
                require(verdict[field] == other[field], f"comparison mismatch: {field}")
            verdict = {"ok": True, "comparison_equal": True,
                       "scientific_gate": verdict["scientific_gate"],
                       "verified_artifact_count": verdict["verified_artifact_count"] + other["verified_artifact_count"],
                       "runs": [verdict, other]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        print(json.dumps({"ok": False, "scientific_gate": None, "verified_artifact_count": 0,
                          "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
