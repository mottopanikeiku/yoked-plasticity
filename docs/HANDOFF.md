# Successor handoff

## Start here

Historical launch checkout: `/home/alp/Projects/github/yoked-plasticity` (not required for reproduction; use your own clone).
Public repository: `https://github.com/mottopanikeiku/yoked-plasticity`.
Sole human author: mottopanikeiku (alp). MIT; main branch. Preserve sole-author Git metadata and do not add coauthor/bot trailers.

**Current research decision: STOP the positive thesis under protocol v1.** Both development configurations executed successfully but failed the joint scientific gate. The one permitted adjustment is consumed. Held-out worlds were not materialized or run. This is a complete reproducible RL launch with a negative initial thesis result, not an unfinished learner and not a claimed breakthrough.

Read, in order:

1. README.md — result and exact entry points.
2. docs/research/THESIS.md — candidate selection, strongest collisions and scope.
3. docs/research/PROTOCOL.md — original design, estimands, gates and budgets.
4. docs/research/DECISIONS.md — A failure, pre-B authorization, final stop.
5. reports/development.md — cells, controls, uncertainty and actual work.
6. docs/research/CLAIMS.md — supported observations versus unestablished claims.
7. docs/research/GRAPH.md and docs/research/LOOPS.md — historical dependencies and bounded iteration.

## What is actually implemented

- Deterministic episodic latent-context control simulator and exact discounted evaluators.
- Float64 NumPy Double DQN with actual Huber/Adam updates, terminal masking and explicit target synchronization.
- Whole-network output-preserving Plasticity Injection, head reset and identity transforms.
- Chronological actor tapes crossed with aged/injected/fresh learner states; absolute replay UIDs; per-update exact diagonal fingerprints.
- Clean uniform-tape comparison including optimizer-only reset; autonomous epsilon/recent-replay controls; an independently learning tabular comparator.
- Aged checkpoint persistence/restoration and complete raw tape/update artifacts.
- Immutable CLI runs, structural-cluster bootstrap summaries, scientific gates and guarded held-out entry.
- Standalone artifact auditing and compute accounting.

There are no model calls, GPU dependencies, external datasets, or fabricated learning scores.

## Environment and commands

Python 3.14; pinned NumPy 2.3.5. Use the repo-local environment, not host package installation:

```sh
uv sync --frozen --python /usr/bin/python3
nice -n 19 uv run --frozen python -m unittest discover -s tests -v
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/sentinel-v2-a --compare results/sentinel-v2-b
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/development-a-v2
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/development-b-v2
nice -n 19 uv run --frozen python tools/summarize_work.py results/development-b-v2
```

To exercise the actual CLI without a full research run:

```sh
nice -n 19 uv run --frozen yoked-plasticity --config configs/smoke.json --output runs/local-smoke
```

That output directory must not already exist. Use a new descriptive name for a later invocation. The CLI fixes BLAS thread counts before NumPy imports.

For an explicit reproduction of frozen development evidence, the commands are:

```sh
nice -n 19 uv run --frozen yoked-plasticity --config configs/development.json --output runs/reproduce-a
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/development-a-v2 --compare runs/reproduce-a
nice -n 19 uv run --frozen yoked-plasticity --config configs/development-b.json --output runs/reproduce-b
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/development-b-v2 --compare runs/reproduce-b
```

Reproduction is an integrity check, not an additional independent sample or permission to search another setting. Exact semantic equality is demonstrated on the launch runtime, not promised across different CPU/BLAS builds. Read runtime versions in the manifests before interpreting a cross-machine mismatch.

## Evidence map

- `results/sentinel-v2-a`, `results/sentinel-v2-b`: repaired-source deterministic sentinels.
- `results/development-a-v2`: configuration A re-executed with the complete initial-score integrity guard.
- `results/development-b-v2`: configuration B re-executed with that same guard.
- Original `sentinel-a/b` and `development-a/b` directories remain immutable historical evidence. Their guard compared later scheduled scores but omitted the step-zero comparison; do not compare their source binding to the repaired current source.
- Each run has exact config bytes, source hashes, environment information, raw per-pair JSONL, checkpoint/tape NPZs, cluster summaries and output hashes.
- `reports/work-a.json`, `reports/work-b.json` and their `-v2` counterparts: reproducible actual work accounting. Development plus integrity re-execution used 6,000,098 training simulator interactions; sentinels are additional.
- `reports/verification.json`: final verification commands and outcomes.

Do not mutate these directories, recalculate their old gate with a new threshold, or remove an inconvenient result. The verifier checks consistency, not an adversarial signature. It hashes NPZ bytes; re-execution of training traces is the runner's job.

## Why the thesis stopped

A showed a material hidden generated-data component but insufficient clean intervention gain and no qualifying reversal. B made the clean intervention response much stronger, but hidden/mixed learner gains had the same sign on both tapes; hidden data allocation was only 0.00080. Existing Tandem RL/AltNet motivation already covers ordinary interaction/data effects. A new algebraic allocation or an injection improvement alone does not clear that collision.

The tabular control reached approximately 0.95/0.98 normalized AUC on hidden/mixed switches. Do not sell this small family as requiring a new complex controller. Three structural clusters also make the intervals exploratory.

## Next action, not an open-ended tuning queue

First verify the preserved artifacts and read the stop decision. **Do not run held-out seeds, add a third development configuration, lower thresholds, or broaden environments under protocol v1.** Graph nodes for locked evaluation, robustness expansion and a positive paper claim are blocked by the failed development prerequisite.

If a future research mandate explicitly reopens this direction, first write a new protocol that explains what new mechanism could escape the Tandem/AltNet collision and why the existing negative result does not settle that different question. It must specify its own cheap falsifier before new learning. Do not silently rebrand that as continuation of the successful launch experiment.

Independent read-only literature work or artifact auditing may run concurrently. One owner writes each new experiment directory; one integrator owns aggregate outputs. Existing unrelated repositories remain untouched. No paid compute is authorized by this handoff.

## Separate exploratory extension

The later [additional-seed plan](seed-check.md) checks the negative result on three more development worlds with configuration B unchanged apart from structural seeds. It does not revise this historical stopping decision, lower thresholds, run the original locked-test worlds, or claim a confirmatory result. New outputs are separate from the launch evidence.
