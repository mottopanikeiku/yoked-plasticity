# Bounded development result: stop the positive thesis

**Both configurations completed numerically and failed the frozen scientific gate.** The one authorized learning-rate adjustment was used. No third configuration and no held-out experiment were run. These results do not support the proposed material tape-dependent intervention reversal.

## Experimental coverage

Each configuration trained six neural checkpoints: three structural worlds × two learner initializations. Each checkpoint produced visible-refit, hidden-opportunity and mixed switches; both autonomous actor tapes were crossed with aged, injected and fresh learners. Additional runs exercised optimizer-only clean-tape calibration, epsilon restart, recent replay and independently aged tabular Q-learning. Exact fingerprints were compared after every diagonal update. Repaired-source v2 runs additionally check the initial score explicitly, so every scheduled diagonal score and final state is verified there.

A: learning rate 0.0003. B: 0.001. Every other field and the entire learning-source binding stayed fixed. A was preserved and its failed clean gate documented before B. The protocol was locally committed before development; this was not an externally registered preregistration.

The single learning_rate controls both aging and adaptation. A-versus-B therefore does not isolate an effect of aging alone.

## Primary effects

Values are means over structural clusters after averaging the two initializations within each cluster. Y is normalized discounted greedy-return AUC; values below zero are permitted. `g_A=Y_IA-Y_AA`, `g_I=Y_II-Y_AI`. Data allocation is the secondary symmetric decomposition, not a natural mediation estimate.

| Run | Switch | Total | Learner allocation | Data allocation | g_A | g_I | Clean I-A | Clean I-O |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Visible | -0.03309 | -0.10971 | 0.07663 | -0.02068 | -0.19874 | -0.01840 | -0.05546 |
| A | Hidden | 0.05470 | -0.02265 | 0.07734 | -0.05230 | 0.00701 | -0.00848 | 0.04682 |
| A | Mixed | 0.13148 | 0.05373 | 0.07775 | 0.04102 | 0.06644 | 0.07971 | 0.05376 |
| B | Visible | 0.01458 | -0.05878 | 0.07336 | -0.07437 | -0.04319 | 0.24372 | 0.16256 |
| B | Hidden | 0.38832 | 0.38752 | 0.00080 | 0.33450 | 0.44054 | 0.39848 | 0.33314 |
| B | Mixed | 0.37433 | 0.35298 | 0.02136 | 0.30259 | 0.40336 | 0.15101 | 0.32714 |

Gate outcomes:

| Run | Clean response | Hidden data | Material conditional reversal | Overall |
| --- | --- | --- | --- | --- |
| A | Fail | Pass | Fail | Stop |
| B | Pass | Fail | Fail | Stop |

A's hidden conditional point estimates have opposite signs, but +0.00701 is far below the prespecified minimum magnitude of 0.05 and the gap is below 0.15. This is not a passing reversal. In B, the higher-rate regime produced a much stronger injection response, but the hidden and mixed conditional effects are positive on both tapes; the desired reversal is absent.

## Uncertainty, not pseudoreplication

There are only **three structural clusters per configuration**, not six independent environments. Bootstrap intervals are exploratory and cannot establish generality or a confirmatory null.

- A, hidden: g_A 95% interval [-0.12347,-0.01643]; g_I [-0.05002,0.04670]; data allocation [0.05233,0.11160]. The positive data signal does not rescue the failed clean/reversal gates.
- B, hidden: g_A [0.03090,0.54308]; g_I [0.35831,0.57798]; data allocation [-0.03509,0.03536]. The negligible data point estimate is consistent with the intervention gain being predominantly a learner-state effect in this configuration. It is not a formal cross-environment equivalence claim.
- B, mixed: g_A [0.09156,0.42778]; g_I [0.33276,0.45379]. Again, both conditional effects point in the same direction.

Every cluster and initialization value, all other intervals, and complete evaluation curves remain in each run's summary.json/results.jsonl.

## Boring baselines

These are actual learned-control AUCs, not source-level wiring checks. The balanced design makes the six-pair mean equal to the structural-cluster mean. Tabular Q-learning is representation-favorable and its aging budget is counted separately.

| Run | Switch | Plain A | Injected I | Epsilon restart | Recent replay | Tabular recent replay |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Visible | 0.28771 | 0.25463 | 0.52156 | 0.37223 | 0.43179 |
| A | Hidden | 0.65057 | 0.70526 | 0.71940 | 0.62476 | 0.95188 |
| A | Mixed | 0.49320 | 0.62468 | 0.48988 | 0.29921 | 0.98272 |
| B | Visible | 0.23281 | 0.24739 | 0.19179 | 0.17736 | 0.43179 |
| B | Hidden | 0.36203 | 0.75035 | 0.33458 | 0.32091 | 0.95188 |
| B | Mixed | 0.42226 | 0.79659 | 0.41367 | 0.43279 | 0.98272 |

The tabular comparator nearly solves hidden/mixed adaptation. The toy family is not evidence that a new controller is needed. Its numbers agree across A and B because the only changed field controls the neural optimizer, not tabular Q-learning. Fresh F-on-each-tape results are retained but cannot diagnose intrinsic plasticity through raw level AUC because F starts with different predictions.

## Actual resource use

| Quantity | A | B |
| --- | ---: | ---: |
| Training simulator interactions | 1,500,023 | 1,500,026 |
| Additional consumer transition reuse | 810,000 | 810,000 |
| Neural optimizer updates | 434,454 | 434,455 |
| Tabular minibatch updates | 142,460 | 142,460 |
| Evaluation simulator interactions | 25,278 | 24,096 |
| Exact-reference transition queries | 648 | 648 |
| Instrumented network-forward examples | 48,604,693 | 48,479,986 |
| Additional initial equivalence-check forward examples, derived from frozen source | 240 | 240 |
| Manifest wall time | 125.13 s | 143.80 s |
| Peak resident memory, process lifetime | 62,554,112 bytes | 62,902,272 bytes |

Initial A/B used **3,000,049 training simulator interactions**, with reused transitions counted separately. Integrity-only v2 re-execution used another 3,000,049, totaling **6,000,098** across development and its verification. No GPU or paid service was used. A network-forward example is one observation processed by one MLP, not a FLOP count. Injection's frozen forwards are included, not treated as free. scripts/summarize_work.py regenerates work-a.json/work-b.json and work-a-v2.json/work-b-v2.json; all scientific work counts match between versions. The table's wall times and memory are from the original runs. Small sentinel runs are additional and excluded from this development total.

## Provenance

Original source binding: `15b0ae34365b1bbca874c8e6c4bf1367c975833a1851a12c697e88d9a743fe0d`. Repaired current-source binding: `79058f47fda74bab472c5d7a4a96523d50224943d7970c39c519c6725d4d9d12`.

- A semantic digest: `76dc1abca488880e2fbfda00d5d8702bfce725b15bc2bc4d00c2ad708e9819f7`.
- B semantic digest: `af6fe0e80b06e4ffdec3075deb23173da59a4a9c1bd99588198b7e6b618fcb35`.
- All four original/repaired sentinel runs: `c00093528cdf7dceda07e9c3f6d9e7fab250fdd4ad4080a1c0a9a9f19bd65a73`.

Independent artifact checks passed for all eight preserved directories. Both A and B have exactly unchanged scientific semantic digests after the step-zero guard repair; the source binding changed, so this is explicitly cross-version integrity revalidation, not an assertion of identical source. The two current-source sentinels also pass strict same-source comparison. See verification.json and review.md. These checks establish internal reproducibility/integrity, not authenticated authorship or scientific novelty.

## Interpretation and stopping consequence

The experiment produced meaningful RL learning, including a substantial output-preserving injection benefit in B. It did **not** produce the stronger discovery needed to distinguish this project from Tandem RL and AltNet. An ordinary injection benefit and a generated-data component in A are already anticipated by the closest literature.

Stop the launch's positive thesis. Keep held-out worlds closed. Do not claim that plasticity interventions never reverse, that intrinsic plasticity was isolated, or that the null generalizes beyond this bounded family. A new scientific direction would need a new explicit protocol and renewed novelty argument, not additional seeds or relaxed gates under this one.
