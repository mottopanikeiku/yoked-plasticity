"""Immutable CPU calibration runs; scientific screening is not numerical success."""

import os

# This module is the entry point: set these before importing NumPy or experiment.
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback


COUNT_FIELDS = (
    "observation_dim", "hidden", "replay_capacity", "batch_size", "update_every",
    "target_every", "age_steps", "block_steps", "adapt_steps", "evaluate_every",
    "restart_steps", "recent_keep",
)
SWITCHES = ("visible_refit", "hidden_opportunity", "mixed")
METRICS = (
    "total_effect", "learner_state_effect", "generated_data_effect", "interaction",
    "clean_gain", "excess_clean_gain",
    "conditional_learner_gain_A", "conditional_learner_gain_I",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
SEED_FIELDS = ("structural_seeds", "learner_seeds")


def validate_config(config):
    """Reject invalid simulator/optimizer inputs before any learning occurs."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    required = set(COUNT_FIELDS) | set(SEED_FIELDS) | {
        "learning_rate", "gamma", "epsilon", "phase", "switches", "intervention",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(sorted(missing))}")
    for name in COUNT_FIELDS:
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["observation_dim"] < 4:
        raise ValueError("observation_dim must be at least 4 for distinct state embeddings")
    if config["age_steps"] % config["block_steps"]:
        raise ValueError("age_steps must contain complete context blocks")
    for name in ("replay_capacity", "recent_keep", "age_steps", "adapt_steps"):
        if config[name] < config["batch_size"]:
            raise ValueError(f"{name} must be at least batch_size")
    for name in ("learning_rate", "gamma", "epsilon"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < config["gamma"] <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not 0 <= config["epsilon"] <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    for name in SEED_FIELDS:
        values = config[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"{name} must contain nonnegative integer seeds")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique seeds")
    if config["phase"] not in ("smoke", "development", "heldout"):
        raise ValueError("phase must be smoke, development, or heldout")
    if config["intervention"] not in ("injection", "head_reset", "identity"):
        raise ValueError("intervention must be injection, head_reset, or identity")
    switches = config["switches"]
    if (not isinstance(switches, list) or not switches
            or any(switch not in SWITCHES for switch in switches)
            or len(set(switches)) != len(switches)):
        raise ValueError("switches must be a nonempty unique list of valid switch names")
    # Also rejects nonfinite numbers in any extension fields before binding them.
    json.dumps(config, allow_nan=False)
    return config


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _without_elapsed(value):
    if isinstance(value, dict):
        return {key: _without_elapsed(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [_without_elapsed(item) for item in value]
    return value


def semantic_digest(value):
    """Bind every scientific value, ignoring only recursively named elapsed_seconds."""
    return hashlib.sha256(_canonical(_without_elapsed(value))).hexdigest()


def _percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _cluster_interval(values):
    # Reinitializing gives the same cluster draws for every paired estimand.
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    if count == 1:
        return [values[0], values[0]]
    draws = sorted(
        math.fsum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def summarize(results, config):
    """Average learner initializations first, then bootstrap structural clusters."""
    if not results:
        raise ValueError("cannot summarize an empty run")
    summary = {
        "schema_version": 1,
        "status": "success",
        "numerical_run_success": True,
        "phase": config["phase"],
        "intervention": config["intervention"],
        "research_status": "preliminary calibration only; not a positive research claim",
        "pair_count": len(results),
        "statistical_unit": "structural_seed; learner initializations averaged within cluster",
        "interval_label": "exploratory" if config["phase"] != "heldout" else "heldout preliminary",
        "bootstrap": {"method": "percentile", "confidence": 0.95,
                      "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "semantic_sha256": semantic_digest(results),
        "switches": {},
    }
    for switch in config["switches"]:
        clustered = {}
        for result in results:
            fields = result["switches"][switch]
            clean = fields["clean_tape_calibration"]
            values = {name: fields[name] for name in METRICS[:4]}
            values["clean_gain"] = clean["I"]["auc"] - clean["A"]["auc"]
            values["excess_clean_gain"] = clean["I"]["auc"] - clean["O"]["auc"]
            cells = fields["cells"]
            values["conditional_learner_gain_A"] = cells["IA"]["auc"] - cells["AA"]["auc"]
            values["conditional_learner_gain_I"] = cells["II"]["auc"] - cells["AI"]["auc"]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
                raise ValueError("scientific summary values must be finite numbers")
            cluster = clustered.setdefault(result["structural_seed"], {})
            learner_seed = result["learner_seed"]
            if learner_seed in cluster:
                raise ValueError("duplicate structural/learner seed pair")
            cluster[learner_seed] = values
        estimates = {}
        for metric in METRICS:
            clusters = []
            for seed, learners in sorted(clustered.items()):
                raw = [{"learner_seed": learner_seed, "value": values[metric]}
                       for learner_seed, values in sorted(learners.items())]
                clusters.append({
                    "structural_seed": seed,
                    "value": math.fsum(row["value"] for row in raw) / len(raw),
                    "learner_seed_values": raw,
                })
            values = [cluster["value"] for cluster in clusters]
            estimates[metric] = {
                "structural_clusters": clusters,
                "structural_cluster_count": len(clusters),
                "mean": math.fsum(values) / len(values),
                "ci95": _cluster_interval(values),
            }
        summary["switches"][switch] = estimates
    means = {switch: {name: estimate["mean"] for name, estimate in estimates.items()}
             for switch, estimates in summary["switches"].items()}
    clean_pass = any(
        means[switch]["clean_gain"] >= 0.10 and means[switch]["excess_clean_gain"] >= 0.05
        for switch in ("visible_refit", "mixed") if switch in means
    )
    hidden = means.get("hidden_opportunity")
    hidden_pass = bool(hidden is not None and hidden["total_effect"] >= 0.05
                       and hidden["generated_data_effect"] >= 0.05
                       and hidden["generated_data_effect"] >= hidden["total_effect"] / 3)
    reversal_switches = [
        switch for switch, values in means.items()
        if values["conditional_learner_gain_A"] * values["conditional_learner_gain_I"] < 0
        and min(abs(values["conditional_learner_gain_A"]), abs(values["conditional_learner_gain_I"])) >= 0.05
        and abs(values["conditional_learner_gain_I"] - values["conditional_learner_gain_A"]) >= 0.15
    ]
    eligible = config["intervention"] == "injection" and config["phase"] != "smoke"
    summary["scientific_gate"] = (
        "advance" if eligible and clean_pass and hidden_pass and reversal_switches else "stop"
    )
    summary["gate_details"] = {
        "name": "frozen preliminary GO gate",
        "eligible_phase_and_intervention": eligible,
        "visible_or_mixed_clean_pass": clean_pass,
        "hidden_opportunity_pass": hidden_pass,
        "conditional_reversal_pass": bool(reversal_switches),
        "reversal_switches": reversal_switches,
        "thresholds": {"clean_gain": 0.10, "excess_clean_gain": 0.05,
                       "hidden_total_effect": 0.05, "hidden_generated_data_effect": 0.05,
                       "hidden_data_fraction_of_total": 1 / 3,
                       "reversal_minimum_absolute_gain": 0.05, "reversal_minimum_gap": 0.15},
        "interpretation": "advance permits the next calibration stage, not a positive research conclusion",
    }
    return summary


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    # Replacements are confined to the new run directory; readers never see torn JSON.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _load_json(path):
    return json.loads(path.read_bytes(), parse_constant=_reject_constant)


def _core_config(config):
    return {name: value for name, value in config.items() if name not in ("phase", *SEED_FIELDS)}


def validate_prerequisite(directory, config):
    """Check a completed, hash-bound development gate before disjoint heldout work."""
    directory = Path(directory).resolve()
    manifest = _load_json(directory / "manifest.json")
    summary = _load_json(directory / "summary.json")
    prior = validate_config(_load_json(directory / "config.json"))
    hashes = _load_json(directory / "output-hashes.json")["files"]
    required = {"manifest.json", "summary.json", "config.json", "results.jsonl"}
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("prerequisite lacks required output hashes")
    actual_files = {str(path.relative_to(directory)) for path in directory.rglob("*")
                    if path.is_file() and path.name != "output-hashes.json"}
    if actual_files != set(hashes):
        raise ValueError("prerequisite output file inventory changed")
    for relative, expected in hashes.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or _sha256_file(path) != expected:
            raise ValueError(f"prerequisite output hash mismatch: {relative}")
    for record in (manifest, summary):
        if (record.get("status") != "success" or record.get("numerical_run_success") is not True
                or record.get("scientific_gate") != "advance"
                or record.get("phase") != "development" or record.get("intervention") != "injection"):
            raise ValueError("prerequisite must be a successful development injection run with gate advance")
    if (directory / "failure.json").exists():
        raise ValueError("prerequisite contains a failure record")
    if prior["phase"] != "development" or prior["intervention"] != "injection":
        raise ValueError("prerequisite configuration is not development injection")
    if manifest.get("config_sha256") != _sha256_file(directory / "config.json"):
        raise ValueError("prerequisite configuration binding mismatch")
    package = Path(__file__).resolve().parent
    current_sources = {f"{package.name}/{path.relative_to(package)}": _sha256_file(path)
                       for path in sorted(package.rglob("*.py"))}
    if manifest.get("source_sha256") != current_sources:
        raise ValueError("heldout learning code must match the frozen development source")
    if _core_config(prior) != _core_config(config):
        raise ValueError("heldout core configuration must exactly match development")
    for name in SEED_FIELDS:
        if set(prior[name]) & set(config[name]):
            raise ValueError(f"heldout {name} overlap prerequisite seeds")
    return {
        "directory": str(directory),
        "manifest_sha256": _sha256_file(directory / "manifest.json"),
        "summary_sha256": _sha256_file(directory / "summary.json"),
        "output_hashes_sha256": _sha256_file(directory / "output-hashes.json"),
    }


def _resources(started):
    # Linux reports KiB, macOS reports bytes. This is process lifetime high water.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"walltime_seconds": time.perf_counter() - started,
            "max_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "max_rss_scope": "process lifetime high-water mark"}


def _print_summary(summary):
    print("switch                 total     learner       data interaction   clean I-A   clean I-O         g_A         g_I", flush=True)
    for switch, estimates in summary["switches"].items():
        print(f"{switch:21s}" + "".join(f" {estimates[name]['mean']:10.5f}" for name in METRICS), flush=True)
    print(f"numerical_run_success=true scientific_gate={summary['scientific_gate']} "
          f"intervals={summary['interval_label']}; preliminary calibration, not a research claim", flush=True)


def run(config_path, output, prerequisite=None):
    """Execute once in an exclusively created directory, recording partial failures."""
    output = Path(output)
    # Outside the exception handler: never write failure metadata into somebody else's run.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"schema_version": 1, "status": "running", "numerical_run_success": False,
                "started_utc": _utc_now(), "command": sys.argv,
                "thread_settings": {name: os.environ[name] for name in THREAD_VARIABLES}}
    completed = 0
    summary = None
    try:
        _write_json(output / "manifest.json", manifest)
        config_bytes = Path(config_path).read_bytes()
        with (output / "config.json").open("xb") as handle:
            handle.write(config_bytes)
        config = validate_config(json.loads(config_bytes, parse_constant=_reject_constant))
        package = Path(__file__).resolve().parent
        source_hashes = {f"{package.name}/{path.relative_to(package)}": _sha256_file(path)
                         for path in sorted(package.rglob("*.py"))}
        manifest.update({
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_sha256": source_hashes,
            "source_binding_sha256": hashlib.sha256(_canonical(source_hashes)).hexdigest(),
            "phase": config["phase"], "intervention": config["intervention"],
            "versions": {"python": sys.version, "python_implementation": platform.python_implementation(),
                         "platform": platform.platform(), "machine": platform.machine()},
        })
        _write_json(output / "manifest.json", manifest)
        print(f"config_sha256={manifest['config_sha256']} "
              f"source_binding_sha256={manifest['source_binding_sha256']} output={output.resolve()}", flush=True)
        if config["phase"] == "heldout":
            if prerequisite is None:
                raise ValueError("heldout phase requires --prerequisite DEV_RUN_DIR")
            manifest["prerequisite"] = validate_prerequisite(prerequisite, config)
        elif prerequisite is not None:
            raise ValueError("--prerequisite is only valid for heldout phase")
        import numpy as np
        from .experiment import run_pair

        manifest["versions"]["numpy"] = np.__version__
        _write_json(output / "manifest.json", manifest)
        results = []
        total = len(config["structural_seeds"]) * len(config["learner_seeds"])
        with (output / "results.jsonl").open("x", encoding="utf-8") as handle:
            for structural_seed in config["structural_seeds"]:
                for learner_seed in config["learner_seeds"]:
                    result = run_pair(structural_seed, learner_seed, config, output)
                    if result["structural_seed"] != structural_seed or result["learner_seed"] != learner_seed:
                        raise ValueError("run_pair returned mismatched seed identifiers")
                    handle.write(_canonical(result).decode("utf-8") + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    results.append(result)
                    completed += 1
                    print(f"pair {completed}/{total}: structural_seed={structural_seed} "
                          f"learner_seed={learner_seed} elapsed_seconds={result.get('elapsed_seconds')}", flush=True)
        summary = summarize(results, config)
        summary.update(_resources(started))
        _write_json(output / "summary.json", summary)
        manifest.update({"status": "success", "numerical_run_success": True,
                         "scientific_gate": summary["scientific_gate"], "completed_pairs": completed,
                         "finished_utc": _utc_now(), "semantic_sha256": summary["semantic_sha256"],
                         **_resources(started)})
        _write_json(output / "manifest.json", manifest)
        hashes = {str(path.relative_to(output)): _sha256_file(path)
                  for path in sorted(output.rglob("*")) if path.is_file()}
        _write_json(output / "output-hashes.json", {
            "algorithm": "sha256", "files": hashes,
            "excludes": ["output-hashes.json"],
        })
        _print_summary(summary)
        return summary
    except BaseException as error:
        failure = {"status": "failed", "numerical_run_success": False,
                   "exception_type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "completed_pairs": completed,
                   "failed_utc": _utc_now(), **_resources(started)}
        # A failed run must never leave a successful manifest, even if finalization failed.
        manifest.update(failure)
        manifest["scientific_gate"] = "stop"
        try:
            _write_json(output / "failure.json", failure)
            _write_json(output / "manifest.json", manifest)
            if summary is not None:
                summary.update({"status": "failed", "numerical_run_success": False, "scientific_gate": "stop"})
                _write_json(output / "summary.json", summary)
        except OSError as recording_error:
            print(f"Could not persist failure record: {recording_error}", file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing paths are never overwritten")
    parser.add_argument("--prerequisite", type=Path, help="successful development run required for heldout")
    args = parser.parse_args(argv)
    run(args.config, args.output, args.prerequisite)


if __name__ == "__main__":
    main()
