# Reproduction notes

I stopped the original positive thesis after both development configurations missed the combined screening criteria. The original held-out worlds were not run. The later [additional-seed check](seed-check.md) and [saved-tape comparison](tape-analysis.md) are separate descriptive analyses; neither changes that stopping decision.

## Environment

I used Python 3.14 and NumPy 2.3.5, pinned in `uv.lock`. The CLI sets BLAS threads to one before importing NumPy. No GPU, external dataset, model call or paid service is needed.

```sh
uv sync --frozen --python 3.14
nice -n 19 uv run --frozen python -m unittest discover -s tests -v
```

For a small execution check, use a new output directory:

```sh
nice -n 19 uv run --frozen yoked-plasticity --config configs/smoke.json --output runs/local-smoke
```

Run directories cannot already exist. Full configuration A and B reproductions use:

```sh
nice -n 19 uv run --frozen yoked-plasticity --config configs/development.json --output runs/reproduce-a
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/development-a-v2 --compare runs/reproduce-a
nice -n 19 uv run --frozen yoked-plasticity --config configs/development-b.json --output runs/reproduce-b
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/development-b-v2 --compare runs/reproduce-b
```

Exact semantic equality was demonstrated on the recorded runtime, not across every CPU or BLAS build. The manifests contain runtime versions. Reproducing a trace is not an independent sample.

On Linux aarch64 (Python 3.14.7, NumPy 2.3.5 wheel, one BLAS thread), `sh scripts/run_seed_check.sh` reproduced every score, AUC, effect, interval, gate, work count and saved tape transition/replay schedule of `results/additional-seeds` exactly, in 221 s. Only the 168 learner-state fingerprints differed: aged checkpoint arrays differ by at most 4e-9, so `verify_artifacts.py --compare` reports a `semantic_sha256` mismatch although every recorded measurement is identical.

## Files behind the results

- `results/development-a-v2` and `results/development-b-v2`: development runs after adding the step-zero diagonal-score check.
- `results/sentinel-v2-a` and `results/sentinel-v2-b`: small deterministic comparisons using that repaired check.
- Earlier `sentinel-a/b` and `development-a/b` directories: preserved runs whose comparison check omitted the initial score. Their source hashes refer to the earlier code.
- `results/additional-seeds`: configuration B with three additional structural seeds, both learner initializations, checkpoints and chronological tapes.
- Each run includes configuration bytes, source hashes, environment information, pair-level JSONL, checkpoint/tape NPZs, cluster summaries and output hashes.
- `reports/development.md` and `reports/additional-seeds.md`: measurements, interpretation and original resource records.
- `reports/work-*.json`: counts of simulator interactions, reused transitions and neural work.

I keep the saved measurements unchanged. The standard-library artifact checker tests file consistency; it does not independently prove learning correctness or authenticate the files. The behavior tests separately check gradients, terminal targets, replay and injection preservation.

```sh
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/additional-seeds
nice -n 19 uv run --frozen python tools/summarize_work.py results/additional-seeds
nice -n 19 uv run --frozen python scripts/analyze_saved_tapes.py results/additional-seeds
```

## Scope

The [original protocol](research/PROTOCOL.md), [decisions](research/DECISIONS.md), [claim discussion](research/CLAIMS.md) and [development report](../reports/development.md) explain the original negative result. Configuration B improved injection response but did not produce an aggregate material reversal; the tabular control nearly solved hidden and mixed switches. I do not interpret this small synthetic family as requiring a new controller, or these exploratory intervals as confirmatory estimates.
