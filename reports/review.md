# Final review and integrity repair

A read-only review inspected the neural update, injection, simulator information boundary, absolute replay UID schedule, crossed cells, controls, statistical clustering and evidence claims.

## Finding: omitted initial diagonal score comparison

The original `consume` implementation recomputed the step-zero score but did not compare it with the actor's recorded step-zero score. It did compare every subsequent scheduled evaluation, every update fingerprint and the final fingerprint. The missing comparison was an integrity-proof gap, not an observed divergence of the completed scientific values.

A regression corrupted only the initial recorded score, leaving all transitions, updates and later scores unchanged. It **failed before the repair** because the consumer incorrectly accepted the tape. The repair adds an explicit initial-score equality assertion before any tape updates. That regression and the other replay/checkpoint tests then passed. The final full suite passed **66 tests**; source distribution and wheel builds also succeeded.

Commit `6d1dbb6` contains the repair and regression. Repaired learning-source binding: `79058f47fda74bab472c5d7a4a96523d50224943d7970c39c519c6725d4d9d12`. Original binding: `15b0ae34365b1bbca874c8e6c4bf1367c975833a1851a12c697e88d9a743fe0d`.

Both original scientific configurations were re-executed under the repaired source in new `development-a-v2` / `development-b-v2` directories. Original artifacts remain intact. A and B each retain exactly the same scientific semantic digest as before the guard repair, and all eight run directories pass artifact auditing. Repaired sentinels also pass strict same-source comparison. Full verdicts are recorded in verification.json. This verification did not introduce a third setting or new independent samples.

## Interpretation limits retained

- One learning_rate controls **both aging and adaptation**. A-versus-B changes cannot be attributed specifically to aging.
- Diagonal equality establishes shared-code deterministic replay. It is not an independent proof that Double DQN is mathematically correct; finite-difference, bootstrap-selection and termination tests provide separate checks.
- Manifests contain package-source hashes and runtime versions, but do not embed a Git revision or copy/hash uv.lock. Those are repository-level context.
- The artifact verifier hashes raw NPZ bytes and checks the recorded diagonal flags; it does not itself re-execute those traces or prove authenticity against wholesale artifact forgery.

The review identified no other causal/learning/replay/control/statistical invalidator in the inspected implementation. That is a scoped review result, not a guarantee of absence of bugs. The scientific stop decision remains separate from engineering correctness.
