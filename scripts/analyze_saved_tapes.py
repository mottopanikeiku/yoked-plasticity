"""Compare saved branch experience and adaptation curves, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from yoked_plasticity.env import Context, World
from yoked_plasticity.experiment import auc


CASES = ((404, "visible_refit"), (606, "mixed"))


def coverage(rows, offset=0):
    """Count root choices, depth visits and successful branch endings."""
    result = []
    for branch in (0, 1):
        start = 1 + 4 * branch
        successes = np.flatnonzero((rows[:, 0] == start + 3) & (rows[:, 2] > 0))
        result.append({
            "branch": branch,
            "entries": int(np.count_nonzero((rows[:, 0] == 0) & (rows[:, 1] == branch))),
            "depth_visits": [int(np.count_nonzero(rows[:, 0] == start + depth)) for depth in range(4)],
            "successes": len(successes),
            "first_success_step": int(successes[0]) + offset + 1 if len(successes) else None,
        })
    return result


def timing(scores):
    """Report sampled optimal crossings; None means no observed crossing."""
    optimal = [abs(score["normalized_return"] - 1.0) <= 1e-12 for score in scores]
    first = next((score["step"] for score, hit in zip(scores, optimal) if hit), None)
    sustained = None
    for index in range(len(scores) - 1, -1, -1):
        if not optimal[index]:
            break
        sustained = scores[index]["step"]
    return {"first_optimal_step": first, "optimal_through_last_sample_from_step": sustained,
            "optimal_sample_count": sum(optimal), "final_normalized_return": scores[-1]["normalized_return"]}


def validate_tape(rows, world, context):
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError("tape must have five finite transition columns")
    state = 0
    for row in rows:
        if row[0] != state or row[1] not in (0, 1):
            raise ValueError("tape has an invalid state or chronological discontinuity")
        next_state, reward, done = world.step(state, int(row[1]), context)
        if row[2] != reward or row[3] != next_state or row[4] != done:
            raise ValueError("tape disagrees with its saved context")
        state = 0 if done else next_state


def analyze(directory):
    config = json.loads((directory / "config.json").read_text())
    records = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    hashes = {}
    for name in ("config.json", "results.jsonl", "summary.json"):
        hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    output = {"source_run": directory.name, "input_sha256": hashes,
              "selection": "Two individual reversals reported earlier; both learner initializations included.",
              "block_steps": 1000, "curve_sample_steps": config["evaluate_every"],
              "original_aggregate_decision": json.loads((directory / "summary.json").read_text())["scientific_gate"],
              "cases": []}
    for seed, switch in CASES:
        selected = sorted((row for row in records if row["structural_seed"] == seed), key=lambda row: row["learner_seed"])
        if [row["learner_seed"] for row in selected] != config["learner_seeds"]:
            raise ValueError("missing or duplicate learner initialization")
        case = {"structural_seed": seed, "switch": switch,
                "anchors_identical": all(row["anchor_context"] == selected[0]["anchor_context"] for row in selected),
                "initializations": []}
        for row in selected:
            anchor = Context(codes=tuple(tuple(code) for code in row["anchor_context"]["codes"]),
                             rewards=tuple(row["anchor_context"]["rewards"]))
            world = World(seed, config["observation_dim"], config["gamma"])
            context = world.post_context(anchor, switch)
            fields = row["switches"][switch]
            cells = {}
            for name in ("AA", "IA", "AI", "II"):
                saved = fields["cells"][name]
                if abs(auc(saved["scores"]) - saved["auc"]) > 1e-12:
                    raise ValueError("saved curve and AUC disagree")
                cells[name] = {"auc": saved["auc"], **timing(saved["scores"]), "scores": saved["scores"]}
            entry = {"learner_seed": row["learner_seed"], "anchor_context": row["anchor_context"],
                     "post_codes": context.codes, "post_rewards": context.rewards,
                     "cells": cells, "g_A": cells["IA"]["auc"] - cells["AA"]["auc"],
                     "g_I": cells["II"]["auc"] - cells["AI"]["auc"], "tapes": {}}
            name = f"tapes-{seed}-{row['learner_seed']}-{switch}.npz"
            hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            with np.load(directory / name, allow_pickle=False) as tapes:
                for source in ("A", "I"):
                    rows = tapes[source + "_transitions"]
                    if len(rows) != config["adapt_steps"]:
                        raise ValueError("tape does not match the adaptation horizon")
                    validate_tape(rows, world, context)
                    entry["tapes"][source] = {"total": coverage(rows), "blocks": [
                        {"start_step": start + 1, "end_step": min(start + 1000, len(rows)),
                         "branches": coverage(rows[start:start + 1000], start)}
                        for start in range(0, len(rows), 1000)]}
            case["initializations"].append(entry)
        case["mean_g_A"] = sum(row["g_A"] for row in case["initializations"]) / len(selected)
        case["mean_g_I"] = sum(row["g_I"] for row in case["initializations"]) / len(selected)
        output["cases"].append(case)
    return output


def fmt(value):
    return "not observed" if value is None else str(value)


def report(result):
    lines = ["# Saved-tape comparison", "", "I compared the two previously reported individual reversals with the other learner initialization in the same structural world. These are descriptive, selected cases, not independent tests. I ran no new learning.", "",
             f"The original aggregate decision remains **{result['original_aggregate_decision']}**. I did not change its criteria.", ""]
    for case in result["cases"]:
        lines += [f"## World {case['structural_seed']}: {case['switch']}", "",
                  f"Both initializations have identical saved anchors: {case['anchors_identical']}. Mean gains within this world: g_A={case['mean_g_A']:.5f}, g_I={case['mean_g_I']:.5f}.", "",
                  "| Initialization | Tape | Branch | Root entries | Depth visits (1–4) | Successes | First success step |", "|---:|:---:|---:|---:|:---|---:|---:|"]
        for row in case["initializations"]:
            for source, tape in row["tapes"].items():
                for branch in tape["total"]:
                    lines.append(f"| {row['learner_seed']} | {source} | {branch['branch']} | {branch['entries']} | {', '.join(map(str, branch['depth_visits']))} | {branch['successes']} | {fmt(branch['first_success_step'])} |")
        lines += ["", "| Initialization | Cell | AUC | First sampled optimal step | Optimal through last sample from | Final normalized return |", "|---:|:---:|---:|---:|---:|---:|"]
        for row in case["initializations"]:
            for name, cell in row["cells"].items():
                lines.append(f"| {row['learner_seed']} | {name} | {cell['auc']:.5f} | {fmt(cell['first_optimal_step'])} | {fmt(cell['optimal_through_last_sample_from_step'])} | {cell['final_normalized_return']:.5f} |")
        lines += ["", "Cells name learner first, tape second: A=aged, I=injected. The JSON includes all original samples and branch counts for each 1,000-interaction block.", ""]
    lines += ["## Interpretation limits", "", "Branch entries count actor experience, not replay minibatch exposure. Successful endings count observed branch rewards, not optimal policy evaluations. First crossings use the saved 250-interaction grid; staying optimal means only at every remaining sampled evaluation. Timing and coverage are associated with performance here, not identified causes. Both initializations share a structural world and anchor, but their aged weights, replay contents and post-switch action streams differ. I did not add an aggregate test or an uncertainty interval for these selected cases.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/tape-comparison.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/tape-comparison.md"))
    args = parser.parse_args()
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.report.write_text(report(result))
    print(args.report)


if __name__ == "__main__":
    main()
