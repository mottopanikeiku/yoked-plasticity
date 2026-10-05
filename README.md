# Yoked Plasticity

This is a small NumPy reinforcement-learning experiment testing whether a plasticity intervention's benefit depends on the experience used to train it.

**Question:** can an intervention help on one learner's experience but hurt on another's?

The [Double DQN implementation](src/yoked_plasticity/learner.py) uses float64 arithmetic and independently implements the published Plasticity Injection method, preserving initial online and target predictions. A [synthetic two-branch task](src/yoked_plasticity/env.py) changes its hidden context during training. The [experiment](src/yoked_plasticity/experiment.py) crosses aged and injected learners with both chronological experience tapes, alongside exploration, replay, optimizer-reset and learned tabular controls.

**Result: the original experiment did not find the material tape-dependent sign reversal it was designed to test.** Injection sometimes helped substantially, but that is not a new algorithm or evidence of isolated intrinsic plasticity.

## Results

The [development report](reports/development.md) contains the original measurements below. Each learning-rate setting used three structural seeds crossed with two learner initializations. AUC is normalized discounted greedy-return area under the adaptation curve; gains are injected minus aged learner AUC on the same tape.

| Setting | Switch | Gain on aged tape | Gain on injected tape | Autonomous injection gain | Tabular AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| A: 0.0003 | Visible | -0.02068 | -0.19874 | -0.03309 | 0.43179 |
| A: 0.0003 | Hidden | -0.05230 | 0.00701 | 0.05470 | 0.95188 |
| A: 0.0003 | Mixed | 0.04102 | 0.06644 | 0.13148 | 0.98272 |
| B: 0.001 | Visible | -0.07437 | -0.04319 | 0.01458 | 0.43179 |
| B: 0.001 | Hidden | 0.33450 | 0.44054 | 0.38832 | 0.95188 |
| B: 0.001 | Mixed | 0.30259 | 0.40336 | 0.37433 | 0.98272 |

A's hidden gains have opposite signs, but the positive gain is too small to meet the original reversal criterion. B's hidden and mixed gains are positive on both tapes. Both settings failed the original combined screening criteria; the original held-out experiment was not run. The learned tabular baseline nearly solves hidden and mixed adaptation.

A [separate exploratory check](reports/additional-seeds.md) added structural seeds 404, 505 and 606 at B's learning rate. Hidden tape-conditioned gains were 0.33251 and 0.45349; tabular hidden/mixed AUCs were 0.97594/0.98702. No aggregate switch met the original reversal criterion, but two individual seed/initialization combinations did. This is heterogeneous evidence, not a general null or a held-out confirmation.

## Reproduce

Use Python 3.14 and [uv](https://docs.astral.sh/uv/); dependencies are pinned in [uv.lock](uv.lock). From the repository root:

```sh
uv sync --frozen --python /usr/bin/python3
nice -n 19 uv run --frozen yoked-plasticity --config configs/development-b.json --output runs/reproduce-b
sh scripts/run_seed_check.sh runs/reproduce-additional-seeds
```

Output directories must be new. Local CPU only; no GPU, external dataset or paid service. Original development settings took about 125–144 seconds each and less than 63 MB peak process memory on the recorded machine ([resource measurements](reports/development.md#actual-resource-use)); runtime varies by CPU. The CLI limits BLAS to one thread. The seed-check script runs the additional configuration and audits its outputs.

Tests: `nice -n 19 uv run --frozen python -m unittest discover -s tests -v`. Historical protocols, decisions and detailed reproduction notes are kept in [docs/HANDOFF.md](docs/HANDOFF.md).

## Limitations

- One small synthetic task family, not an Atari or real-world benchmark.
- Only three structural clusters per original setting; uncertainty intervals are exploratory.
- The learning rate affects aging and adaptation, so A/B does not isolate aging.
- Tabular control has a favorable representation; neither its success nor neural failure establishes a general ranking.
- Shared-code replay consistency is not independent algorithm verification; numerical gradient and terminal-target tests provide separate checks.

## Prior work and attribution

[Tandem RL](https://arxiv.org/abs/2110.14020) motivates separating data generation from passive learning. [Plasticity Injection](https://arxiv.org/abs/2305.15555) supplies the intervention. [AltNet](https://arxiv.org/abs/2512.01034) studies reset passive networks on active trajectories and replay interactions. These methods are independently implemented, not invented here; no upstream implementation is vendored. MIT license; author: [Alp Cetin](https://github.com/mottopanikeiku). See [NOTICE.md](NOTICE.md).
