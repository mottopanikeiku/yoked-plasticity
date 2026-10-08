# Yoked Plasticity

**I found two individual material tape-dependent injection reversals, but neither met the unchanged material criterion after averaging both learner initializations in its structural world.** The original aggregate experiment still stops. In the saved-tape follow-up, one injected actor observed **zero** rewards from the optimal branch, while another supplied **517** such rewards and still trained its aged consumer faster than its injected consumer ([comparison](reports/tape-comparison.md), [interpretation](docs/tape-analysis.md)). Coverage alone does not explain learning.

This is a small NumPy reinforcement-learning experiment asking whether an intervention can help on one learner's experience but hurt on another's. I independently implemented [Plasticity Injection](https://arxiv.org/abs/2305.15555), not a new intervention. The [float64 Double DQN](src/yoked_plasticity/learner.py) preserves initial online and target predictions at injection. A [synthetic two-branch world](src/yoked_plasticity/env.py) changes context; the [experiment](src/yoked_plasticity/experiment.py) crosses aged and injected learners with both chronological actor tapes, alongside exploration, replay, optimizer-reset and learned tabular controls.

## What the saved tapes show

I compared both initializations in the two worlds containing the previously reported individual reversals. Gains are injected-minus-aged normalized return AUC on the same tape; A is aged actor experience and I is injected actor experience.

| World / switch | Initialization | Gain on A tape | Gain on I tape |
|---|---:|---:|---:|
| 404 / visible | 11 | +0.39789 | +0.41078 |
| 404 / visible | 22 | -0.34121 | +0.20310 |
| 606 / mixed | 11 | +0.46913 | -0.15551 |
| 606 / mixed | 22 | -0.44099 | -0.72658 |

Sources: [machine-readable comparison](results/tape-comparison.json) and [saved curves and coverage tables](reports/tape-comparison.md). The script reconstructs each pair's actual context and checks all eight tapes against the simulator. Both initializations share the same anchor within each world.

In world 404/22, the aged actor completed the optimal branch 361 times; its injected counterpart never completed it. Only the aged learner on aged experience reached an optimal sampled return, from interaction 3,250 onward. In world 606/11, the injected tape completed the optimal branch 517 times, versus four on the aged tape. Yet the aged consumer of injected experience first reached the optimum at 1,500, compared with 2,250 for the injected consumer. Coverage and learner state differ together; these selected observations do not identify their causal contributions.

## Original result remains negative

The [original development report](reports/development.md) found no material aggregate reversal at either learning rate. At configuration B's rate, the [additional three-world check](reports/additional-seeds.md) had hidden tape-conditioned gains of 0.33251/0.45349 and tabular hidden/mixed AUCs of 0.97594/0.98702. No aggregate switch passed the original criterion: opposite signs, both absolute gains at least 0.05 and a gap of at least 0.15, after averaging initializations within structural clusters. I did not change those criteria or run the original held-out worlds.

## Reproduce without training

Python 3.14 and pinned NumPy; local CPU, no GPU or paid service:

```sh
uv sync --frozen --python 3.14
nice -n 19 uv run --frozen python scripts/analyze_saved_tapes.py results/additional-seeds
nice -n 19 uv run --frozen python -m unittest discover -s tests -v
```

The analysis regenerates the JSON and table from saved data, not new learning or timing measurements. [Reproduction notes](docs/HANDOFF.md) explain full training commands, artifact checks and runtime versions. [The analysis plan](docs/tape-analysis.md) records descriptive metrics chosen before computing the new summaries.

## Limitations

- One small synthetic task family, with three structural clusters per development setting.
- The follow-up selects two previously observed cases; it is not independent confirmation or a general null.
- Timing uses sampled greedy returns every 250 interactions, not continuous behavior; reward observations are not replay exposure counts.
- Initializations differ in weights, inherited replay and action streams. Shared-code consistency is not independent RL verification.
- Tabular control has a favorable representation; its success does not establish a general learner ranking.

## Prior work

[Tandem RL](https://arxiv.org/abs/2110.14020) motivates separating data generation from passive learning. [Plasticity Injection](https://arxiv.org/abs/2305.15555) supplies the intervention. [AltNet](https://arxiv.org/abs/2512.01034) studies reset passive networks and replay interactions. No upstream code is vendored; see [NOTICE.md](NOTICE.md). MIT license.

Written with AI coding assistance.
