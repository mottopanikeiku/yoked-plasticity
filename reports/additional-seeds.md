# Additional structural-seed result

The separately planned exploratory check did **not** meet the original material reversal criterion on structural-cluster means. It did expose individual seed/initialization reversals, so the result is not an absence of reversals in every run.

## Design

The [plan](../docs/seed-check.md) was recorded before the run. [Configuration B](../configs/development-b.json) was unchanged except for structural seeds 404, 505 and 606; each was crossed with learner seeds 11 and 22. This is an additional development sample, not the original protocol's held-out test and not a tuning attempt. The original stopping decision remains unchanged.

Full checkpoints, chronological tapes, per-pair curves, source/configuration hashes and structural-cluster summaries are in [results/additional-seeds](../results/additional-seeds). The learner code was unchanged from the repaired development runs.

## Aggregate measurements

Means average the two learner initializations within each structural seed before averaging the three clusters. `g_A` and `g_I` are injected-minus-aged learner normalized adaptation AUC on the aged and injected tapes, respectively. Intervals are exploratory percentile bootstrap intervals over just three structural clusters, not confirmatory confidence statements. Tabular means are computed from the six raw control AUCs in `results.jsonl`; the balanced design gives the same cluster mean.

| Switch | g_A (95% interval) | g_I (95% interval) | Autonomous injection gain | Data allocation | Tabular AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visible | -0.02128 [-0.23598, 0.14381] | 0.08992 [-0.29915, 0.30694] | 0.00933 | -0.02498 | 0.46365 |
| Hidden | 0.33251 [0.03396, 0.52649] | 0.45349 [0.06037, 0.66403] | 0.30813 | -0.08488 | 0.97594 |
| Mixed | 0.43783 [0.01407, 0.88058] | 0.39626 [-0.44104, 0.89982] | 0.56073 | 0.14369 | 0.98702 |

The original criterion requires opposite signs, both absolute gains at least 0.05, and a gap of at least 0.15 on the switch's aggregate means. Visible has opposite signs but its negative gain is too small and its gap is only 0.11119. Hidden and mixed means have the same sign. The existing CLI reports clean-response pass, hidden-data fail and reversal fail, with overall decision `stop`. Hidden's data allocation is negative, despite a positive autonomous injection gain. The learned tabular comparator again nearly solves hidden and mixed adaptation.

## Heterogeneity matters

Two individual seed/initialization combinations satisfy those magnitude and gap thresholds when considered alone:

| Structural seed | Learner seed | Switch | g_A | g_I |
| ---: | ---: | --- | ---: | ---: |
| 404 | 22 | Visible | -0.34121 | 0.20310 |
| 606 | 11 | Mixed | 0.46913 | -0.15551 |

These are descriptive observations found after inspecting the results, not independent aggregate tests or a reason to relax the planned criterion. For seed 606's mixed switch, averaging its two learner initializations gives g_A=0.01407 and g_I=-0.44104; that cluster still misses the minimum positive magnitude. The small sample supports neither a general null nor a reliable intervention ranking.

## Execution and checks

Executed once with:

```sh
nice -n 19 uv run --frozen yoked-plasticity --config configs/additional-seeds.json --output results/additional-seeds
nice -n 19 uv run --frozen python tools/verify_artifacts.py results/additional-seeds
nice -n 19 uv run --frozen python tools/summarize_work.py results/additional-seeds --output reports/work-additional-seeds.json
```

All six pairs completed. The standard-library artifact audit passed all 28 recorded artifacts; this checks consistency, not independent learning correctness. The separate behavior suite passed 66 tests in 1.838 seconds, including finite-difference gradients, injection preservation, replay corruption and held-out blocking.

The [manifest](../results/additional-seeds/manifest.json) records Python 3.14.7, NumPy 2.3.5, Linux x86-64, one BLAS thread, 149.75 seconds wall time and 62,013,440 bytes peak resident memory. The [work counts](work-additional-seeds.json) record 1,500,023 training simulator interactions and 810,000 additional consumer transition reuses. Local CPU only; $0 paid compute. Available memory before the run was 2,962 MiB.

For reproduction after `uv sync --frozen`, run `sh scripts/run_seed_check.sh runs/reproduce-additional-seeds`. The script uses a two-hour timeout and a new output directory, then audits the run and prints its work counts. The wrapper itself was not executed; its component experiment, audit and accounting commands above were executed.

## Next step

Before spending more compute, study why the two individual reversals disappear or change sign across initializations using the already saved tapes and learning curves. A useful analysis would compare branch coverage and adaptation timing across both learner seeds within those structural worlds, without selecting a new aggregate criterion after the fact. Any new harder-task experiment needs its own question and stopping rule; no further run or held-out evaluation is claimed here.

The reproduction wrapper passed `sh -n scripts/run_seed_check.sh` (shell syntax only; no additional training run).
