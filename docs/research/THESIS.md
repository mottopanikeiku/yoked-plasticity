# Yoked Plasticity: tape-dependent intervention reversals

## Decision

Study a narrow continual-reinforcement-learning question: **does an initially output-preserving plasticity intervention help on one learner's generated experience but hurt on the other's?** A reproducible reversal would make a one-way passive-learning comparison insufficient for ranking that intervention. An ordinary exploration benefit, a four-cell table, or a Shapley decomposition is not a research contribution.

This is a conditional research launch, not a demonstrated significant result. The initial CPU experiment can stop the thesis. Do not treat the project name, documentation, or implementation as evidence of novelty.

## Why this question

RL learners change the experience from which they learn. Freezing an initial prediction function does not freeze the future behavior induced by different gradients. A reset can therefore have a tape-dependent learning effect even when its initial online and target predictions are unchanged. The proposed discovery is a material *change of sign* in that effect, not the already-known importance of interaction and coverage.

The experimental objects are real Double-DQN learners, chronological environment interactions, replay minibatches, and learned control returns. There is no policy oracle in training and no synthetic score standing in for learning.

## Closest work: strong collision, narrow residual

- [Tandem RL / The Difficulty of Passive Learning in Deep Reinforcement Learning (2021)](https://arxiv.org/abs/2110.14020) already decouples learning from data generation, gives active/passive agents identical ordered batches, uses Double DQN, studies coverage, and includes forked identical initializations. This is the foundational predecessor, not a peripheral citation.
- [Plasticity Injection (2023)](https://arxiv.org/abs/2305.15555) already supplies the output-preserving residual intervention and explicitly recognizes the exploration confound. Equation 1 is the intervention used here with the whole MLP as the head; this is not a reproduction of its Atari encoder/head architecture or Atari results.
- [AltNet (AAMAS 2026)](https://arxiv.org/abs/2512.01034) already trains reset passive networks on active trajectories, alternates their roles, and studies resets jointly with replay preservation. Its discussion acknowledges exploration. A same-data reset advantage would merely reproduce its motivation.
- [Revisiting Plasticity in Visual RL (2024)](https://arxiv.org/abs/2310.07418) includes data-augmentation/reset interactions and actor/critic intervention probes. Factorial analysis in this area is not new in general.
- [Resetting the Optimizer in Deep RL (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html) makes optimizer reset a substantive alternative explanation. The clean-tape experiment includes it explicitly.
- [Primacy Bias (2022)](https://proceedings.mlr.press/v162/nikishin22a.html), [ReDo (2023)](https://proceedings.mlr.press/v202/sokar23a.html), and [Loss of Plasticity (2024)](https://www.nature.com/articles/s41586-024-07711-7) establish the wider problem and major remedies. This project does not introduce a new remedy.

The remaining question is whether crossing both learner states with both endogenous tapes reveals a consequential conditional reversal not identifiable from either single one-way control. Targeted literature review did not establish priority. The hostile novelty assessment was **conditional, leaning kill**. A paper-level novelty claim requires a renewed comparison with Tandem RL and AltNet plus evidence beyond this toy family.

## Candidate comparison

Five independently researched RL directions were considered before selection:

| Direction | Surviving question | Reason not selected |
| --- | --- | --- |
| Reset-minimizing continuing RL | Value of practicing recovery before spending an external reset | Budget-augmented optimistic/Bayesian RL may already internalize the learning value; must beat a strong dual-control composition. |
| Offline latent-context RL | Passive estimation of higher-order within-episode co-coverage for relative policy improvement | ARMOR, latent-MDP coverage results, and pessimistic POMDP learning leave a fragile identifiability/estimation gap; a null result is likely. |
| Long-horizon credit assignment | Policy-conditioned hindsight credit under continuation-policy sign drift | COCOA, policy fingerprints, and policy-conditioned credit are close; generic conditioning is not new. |
| Action redundancy/geometry | Reversible finite-sample action equivalence for control | Unknown-equivalence UCRL, Bellman-compatible abstractions, and online action refinement substantially narrow novelty. |
| Continual RL: selected | Tape-conditional intervention sign reversal | Strong collision too, but a complete falsifiable CPU mechanism experiment is feasible without a new opaque controller or paid compute. |

Selection reflects a defensible question and a cheap way to reject it, not confidence that it will succeed or a proof of global optimality.

## Boundaries

- No claim of discovering the exploration confound, inventing Plasticity Injection, or outperforming AltNet.
- A learner-state contrast is not automatically intrinsic neural plasticity. It includes initialization geometry and optimizer state; no early-versus-aged matched-function comparison is performed in this launch.
- A fixed structural seed supplies a small nonstationary latent-context control family. Results do not generalize to Atari, robotics, recurrent policies, or arbitrary continual RL.
- No natural causal mediation claim. The four cells are controlled algorithmic outcomes under specified tapes and update schedules.
- Existing repositories and research theses remain untouched. This project is neither active causal diagnosis nor an RL distillation/FHE project.

## Success and failure

The locally frozen protocol is in [PROTOCOL.md](PROTOCOL.md). Initial development must satisfy clean-tape, endogenous-data, and conditional-reversal gates before held-out execution is permitted. Failure is retained as evidence, not repaired by changing thresholds or opening held-out seeds. The repository must remain useful as a reproducible negative launch if the thesis fails.
