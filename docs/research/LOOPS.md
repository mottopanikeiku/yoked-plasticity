# Research loops

Every loop records **goal → action → observation → adjustment**. A new run is a new immutable directory. The goal is to resolve a causal question, not obtain a passing number.

## Launch loops

| Loop | Goal | Action | Observation | Adjustment/terminal rule |
| --- | --- | --- | --- | --- |
| Literature | Find a distinct, consequential RL question | Five independent direction reviews, then hostile novelty/design review | Tandem RL and AltNet own most of the selected motivation | Narrow to material tape-conditional intervention reversals; explicitly conditional novelty |
| Design | Make the proposed contrast interpretable | Replace head reset as primary with output-preserving injection; add optimizer-only and clean-tape controls; fix discount/statistical units/replay capacity before development | A four-cell result can identify an algorithmic contrast, not intrinsic plasticity or natural mediation | Freeze the precise claim boundary and all development gates |
| Numerical integrity | Establish actual learning and replay correctness | Behavioral numerical tests, checkpoint continuation, corrupted-tape rejection, identity control, two full CLI sentinels | Results recorded in reports/verification.json | Any integrity failure blocks scientific interpretation; diagnose before changing code |
| Development A | Test clean response, hidden-data effect and conditional reversal jointly | Six paired runs from configs/development.json | A completes numerically but fails clean and reversal gates | Preserve A; the protocol permits one LR-only B because clean calibration failed |
| Development B | Test the reserved higher-rate aging hypothesis | Same six pairs with only learning_rate changed to 0.001 | Final observation recorded in DECISIONS.md and reports/development.md | All gates pass: consider locked eligibility. Any gate fails: stop this launch's positive thesis |

## Failure taxonomy

1. **Engineering/integrity failure:** exception, nonfinite arithmetic, overwritten replay UID, diagonal divergence, incomplete seed grid, or failed artifact hash. No scientific sign conclusion is valid. Preserve partial outputs and failure metadata. A diagnosed transient may receive at most one exact rerun; a causal code repair gets its own source binding and cannot masquerade as an exact rerun.
2. **Scientific failure:** numerically valid data do not meet a prespecified gate. This is an observation, not a broken test. Do not rerun another random seed hoping for a pass.
3. **Novelty failure:** the observed effect is already explained by closest work or by a trivial composition. More engineering does not repair this. Stop or propose a genuinely new, explicitly scoped protocol.

## Change discipline

- One causal hypothesis per adjustment. Write its reason and exact changed field before execution.
- Never change a threshold, seed list, environment family, budget and optimizer simultaneously.
- The launch permits A and, only under its stated condition, B. There is no third configuration and no implicit sweep.
- Tests and sentinels are cheaper than development; development is cheaper than locked evaluation. Use that order.
- The held-out phase is not a debugging tool. Failed prerequisites keep it closed.
- Reading a stopped run to understand failure is permitted. Presenting exploratory post-hoc contrasts as frozen primary results is not.
- A stop is a complete scientific branch: publish enough negative evidence for the successor not to repeat it unknowingly.

## Evidence retention

`results/*/config.json` preserves exact inputs; `manifest.json` separates numerical success from the scientific gate; `results.jsonl` retains each completed pair; NPZ files preserve tapes/checkpoints; `summary.json` reports structural-cluster estimates; `output-hashes.json` binds the inventory. Exceptions retain `failure.json` and already-flushed rows. Never overwrite an old directory or manually change a recorded gate.

The artifact verifier checks internal consistency and deterministic sentinel equivalence. It is not an adversarial signature, novelty certificate, or substitute for reading the four raw cells. Sole-writer ownership and downstream claim dependencies are in GRAPH.md.

## Closed engineering repair loop

Goal: make the claimed diagonal check cover the entire scheduled curve. Action: corrupt only the actor's initial recorded score. Observation: the original guard incorrectly accepted it. Adjustment: add the missing initial-score comparison; the new regression and final 66-test suite pass.

Both existing configurations were then re-executed once under the repaired source, in new `-v2` directories. Their complete scientific semantic digests exactly match the originals. This separate integrity branch changed no learner, setting, seed or threshold and created no independent scientific samples. The failed scientific gates stay failed.
