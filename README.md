# Yoked Plasticity

**A continual-reinforcement-learning study of tape-dependent intervention reversals.**

Question: can an initially output-preserving plasticity intervention help when trained on one learner's generated experience but hurt on the other's? The experiment crosses actual Double-DQN learner states with both chronological behavior tapes, rather than attributing every autonomous return gain to restored plasticity.

**Launch result: the proposed positive thesis did not survive the bounded development protocol.** Both six-pair configurations completed numerically; neither passed the required material sign-reversal gate. The one permitted learning-rate adjustment was used. Held-out evaluation remains closed. This repository preserves the implementation, raw evidence and exact stopping decision—not a claim of a new RL breakthrough.

## Result in brief

Three structural worlds × two initializations per configuration; visible, hidden-opportunity and mixed switches. Means use structural clusters, with exploratory uncertainty rather than treating every rollout as independent.

| Configuration | Clean response gate | Hidden-data gate | Reversal gate | Decision |
| --- | --- | --- | --- | --- |
| A: learning rate 0.0003 | Fail | Pass | Fail | Stop |
| B: learning rate 0.001 | Pass | Fail | Fail | Stop |

B produced a substantial hidden-opportunity injection benefit: total normalized adaptation-AUC gain **0.38832**, with learner-state allocation **0.38752** and generated-data allocation **0.00080**. That is meaningful learning evidence, but not the new tape-dependent reversal the thesis required. Initial A/B used **3,000,049 training simulator interactions**; integrity re-execution used the same amount, for **6,000,098** across those runs, CPU only. Small sentinels and consumer replay reuse are accounted separately.

Read the [full report](reports/development.md), [claim ledger](research/CLAIMS.md), and [decision history](research/DECISIONS.md). The negative development result does not establish a universal null.

## What runs

- A procedural, hidden-context sequential-control environment with two actions and five-decision episodes.
- Float64 NumPy Double DQN: two 64-unit ReLU layers, Huber TD loss, Adam, target network and epsilon-greedy exploration.
- Whole-network output-preserving [Plasticity Injection](https://arxiv.org/abs/2305.15555), preserving both initial online and lagged target functions.
- Complete aged/injected learner × aged/injected actor-tape crossing, plus fresh learners on each tape.
- Optimizer-reset-only clean-tape calibration; epsilon restart; rolling recent replay; independently learned tabular control.
- Exact discounted control evaluation, absolute replay UIDs, per-update diagonal fingerprints, serialized checkpoints and raw tapes.
- Immutable runs, cluster-level summaries, artifact auditing and reproducible compute accounting.

Only observations and ordinary transitions reach the learner. Context codes, oracle policies and evaluation results never enter replay. This is actual RL learning, not a systems benchmark with RL terminology.

## Quick start

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). No GPU, model endpoint, paid API or external dataset.

```sh
uv sync --frozen --python /usr/bin/python3
uv run --frozen yoked-plasticity --config configs/smoke.json --output runs/my-smoke
uv run --frozen python -m unittest discover -s tests -v
```

The CLI refuses an existing output directory. Use a new directory for each execution. The smoke configuration exercises real learning and every crossover/control path, but cannot authorize a scientific advance.

Audit the published evidence without retraining:

```sh
uv run --frozen python scripts/verify_artifacts.py results/sentinel-v2-a --compare results/sentinel-v2-b
uv run --frozen python scripts/verify_artifacts.py results/development-a-v2
uv run --frozen python scripts/verify_artifacts.py results/development-b-v2
uv run --frozen python scripts/summarize_work.py results/development-b-v2
```

Exact development reproduction commands and runtime caveats are in [HANDOFF.md](HANDOFF.md). Reproduction does not authorize another search configuration or create independent scientific samples.

## Novelty boundary

[Tandem RL (2021)](https://arxiv.org/abs/2110.14020) already separates active data generation from passive learning with identical ordered batches. [Plasticity Injection (2023)](https://arxiv.org/abs/2305.15555) already supplies the intervention and recognizes exploration confounding. [AltNet (2026)](https://arxiv.org/abs/2512.01034) already trains reset passive networks on active trajectories and studies replay/reset interactions.

The intended residual contribution was a consequential tape-conditional sign reversal that a one-way comparison cannot reveal. A four-cell table, a symmetric allocation, ordinary coverage effects, or an injection benefit alone are not novel contributions. The initial experiment did not clear that boundary. See [THESIS.md](research/THESIS.md) for the candidate comparison and hostile-review conclusions.

## Research operating model

- [Protocol](research/PROTOCOL.md): frozen estimands, seed split, gates, controls and bounded adjustment.
- [Dependency graph](research/GRAPH.md): sole writers, exact inputs/outputs, acceptance and stop propagation.
- [Research loops](research/LOOPS.md): goal → action → observation → one justified adjustment.
- [Successor handoff](HANDOFF.md): executable baseline, evidence map and explicit closed branches.

Numerical success is separate from scientific acceptance. Failed prerequisites block downstream positive claims. Current status: preserve the negative result; do not open held-out worlds or continue tuning under protocol v1.

## License and authorship

MIT. Sole human author: **mottopanikeiku (alp)**. Published algorithms and dependencies retain their attribution; see [NOTICE.md](NOTICE.md) and [CITATION.cff](CITATION.cff).
