# Saved-tape comparison

## Analysis plan

I will compare the two previously reported individual reversals: visible refit in structural world 404 (learner initialization 22) and the mixed switch in world 606 (initialization 11). For each world and switch I will include both saved learner initializations, both actor tapes, and all four aged/injected learner–tape combinations. These cases were selected from the earlier result, not as new independent tests.

Before reading the new summaries, I fixed the descriptive measurements: branch entries at the root; visits to every depth; successful terminal reward observations per branch; first success interaction; and the same counts in each 1,000-interaction block. I will compare all saved normalized-return curves on their original 250-interaction grid, including AUC, the first sampled optimal return and the first sampled return that stays optimal through the last evaluation. A missing crossing will remain missing, not be replaced by the horizon. Sampling limits timing resolution, and observing a reward is not proof that replay trained on it.

I will reconstruct the actual post-switch context from each pair's saved anchor and check the tapes against the simulator. If the anchors differ across initializations, I will state that limitation instead of treating the comparison as identical tasks. The original aggregate criterion remains opposite signs, both absolute gains at least 0.05, and a gap of at least 0.15 on structural-cluster means. I will neither change that criterion nor run training or held-out worlds.

## Reproduction

The comparison uses the saved files under `results/additional-seeds/`; no new learning or timing measurement is needed. Run `nice -n 19 uv run --frozen python scripts/analyze_saved_tapes.py results/additional-seeds` after installing the locked environment. It writes [all measurements](../results/tape-comparison.json) and [the tables](../reports/tape-comparison.md); input hashes bind the saved configuration, curves, original summary and eight tapes.

## What I found

World 404's visible switch refits branch 1, whose reward is 1.0 rather than branch 0's 0.2. At initialization 22, the aged tape completed branch 1 361 times (first at interaction 1,278), while the injected tape completed it zero times despite 243 root entries. The aged learner on its own tape reached an optimal sampled return at 3,250 and stayed there through the last sample; neither consumer of injected experience ever reached the optimum. Injection's positive gain on that tape is improvement to the lower-reward branch, not successful refitting of the optimal branch.

At initialization 11, the aged tape completed branch 1 only 29 times and the injected tape 253 times. The aged consumers never reached an optimal sampled return, while both injected consumers did. Their AUC gains are positive on both tapes (+0.39789/+0.41078), unlike initialization 22 (-0.34121/+0.20310). Averaging the two gives +0.02834/+0.30694, so this world no longer has opposite signs.

World 606's mixed switch makes branch 0 optimal (reward 1.5 rather than branch 1's 0.2). At initialization 11, the aged tape observed four optimal-branch rewards, all in the first 1,000 interactions. The injected tape observed 517, with 157, 145 and 163 in the last three blocks. The injected learner on aged experience reached an optimal sampled return at 1,500 but did not retain it at the final sample. On injected experience, the aged learner reached and retained the sampled optimum from 1,500; the injected learner did so only from 2,250. Thus a well-covered optimal branch does not imply an advantage for injection on that experience.

At initialization 22, optimal-branch reward counts were 286/218 on aged/injected tapes; gains were negative on both (-0.44099/-0.72658). The aged consumer of injected experience first reached an optimal sampled return at 500; the injected consumer did so at 4,250. Their final returns were both optimal, so the large AUC gap reflects adaptation timing rather than different final sampled performance. Averaging the two initializations gives +0.01407/-0.44104: opposite signs remain, but the positive magnitude misses the unchanged 0.05 threshold.

I can describe the heterogeneity more precisely, but I cannot attribute it solely to branch coverage. Same-world initializations share anchors yet differ in aged weights, inherited replay and action streams. In particular, consumer performance need not track its actor's chronological success count because it also learns from inherited replay. These selected comparisons explain where the observed reversals occur; they do not make a new aggregate claim.
