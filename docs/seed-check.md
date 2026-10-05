# Additional structural-seed check

## Plan recorded before running

This is a separate exploratory extension of the negative development result, not another attempt to pass the original protocol or a held-out confirmation. The original A/B runs and stopping decision remain unchanged.

Use configuration B (learning rate 0.001) on three additional structural seeds, 404, 505 and 606, crossed with the same learner seeds, 11 and 22. Keep the task, aging and adaptation budgets, controls and thresholds unchanged. These are manually selected additional development worlds, not the original protocol's derived locked-test worlds. Do not pool them with the original clusters or tune after seeing them.

The question is whether the missing material tape-dependent sign reversal persists on this small additional sample. Report both conditional injection gains, their exploratory structural-cluster intervals, the existing reversal criterion (opposite signs, each magnitude at least 0.05, gap at least 0.15), and the learned tabular comparator. Passing any historical screening threshold here would not reopen the original stopped study.

Run once on local CPU under `nice -n 19`, with one BLAS thread, no paid services, and a two-hour deadline. Preserve the full output directory, including a failure if one occurs. Reproduction uses `scripts/run_seed_check.sh`; use a new output directory.

## Result

The run completed all six pairs without changing settings afterward. Aggregate means again missed the material reversal criterion, but two individual seed/initialization combinations met its sign and magnitude conditions. The [result report](../reports/additional-seeds.md) records this heterogeneity, tabular controls, exploratory intervals and actual resource use. The original stopped study remains stopped.
