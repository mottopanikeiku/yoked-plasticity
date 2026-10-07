# Saved-tape comparison

I compared the two previously reported individual reversals with the other learner initialization in the same structural world. These are descriptive, selected cases, not independent tests. I ran no new learning.

The original aggregate decision remains **stop**. I did not change its criteria.

## World 404: visible_refit

Both initializations have identical saved anchors: True. Mean gains within this world: g_A=0.02834, g_I=0.30694.

| Initialization | Tape | Branch | Root entries | Depth visits (1–4) | Successes | First success step |
|---:|:---:|---:|---:|:---|---:|---:|
| 11 | A | 0 | 596 | 596, 523, 486, 388 | 372 | 9 |
| 11 | A | 1 | 805 | 805, 733, 37, 31 | 29 | 367 |
| 11 | I | 0 | 633 | 633, 533, 476, 450 | 430 | 5 |
| 11 | I | 1 | 627 | 627, 448, 308, 265 | 253 | 274 |
| 22 | A | 0 | 595 | 595, 536, 507, 41 | 40 | 20 |
| 22 | A | 1 | 648 | 648, 600, 428, 402 | 361 | 1278 |
| 22 | I | 0 | 977 | 977, 853, 814, 766 | 719 | 20 |
| 22 | I | 1 | 243 | 243, 86, 29, 12 | 0 | not observed |

| Initialization | Cell | AUC | First sampled optimal step | Optimal through last sample from | Final normalized return |
|---:|:---:|---:|---:|---:|---:|
| 11 | AA | 0.06646 | not observed | not observed | 0.21007 |
| 11 | IA | 0.46435 | 2000 | 4250 | 1.00000 |
| 11 | AI | -0.03985 | not observed | not observed | -0.03989 |
| 11 | II | 0.37094 | 500 | 4250 | 1.00000 |
| 22 | AA | 0.35089 | 3250 | 3250 | 1.00000 |
| 22 | IA | 0.00969 | not observed | not observed | -0.04151 |
| 22 | AI | -0.03698 | not observed | not observed | -0.03679 |
| 22 | II | 0.16612 | not observed | not observed | 0.21007 |

Cells name learner first, tape second: A=aged, I=injected. The JSON includes all original samples and branch counts for each 1,000-interaction block.

## World 606: mixed

Both initializations have identical saved anchors: True. Mean gains within this world: g_A=0.01407, g_I=-0.44104.

| Initialization | Tape | Branch | Root entries | Depth visits (1–4) | Successes | First success step |
|---:|:---:|---:|---:|:---|---:|---:|
| 11 | A | 0 | 210 | 210, 161, 7, 6 | 4 | 635 |
| 11 | A | 1 | 1016 | 1016, 907, 754, 713 | 656 | 542 |
| 11 | I | 0 | 705 | 705, 656, 592, 550 | 517 | 30 |
| 11 | I | 1 | 579 | 579, 419, 169, 46 | 28 | 516 |
| 22 | A | 0 | 804 | 803, 710, 345, 313 | 286 | 2306 |
| 22 | A | 1 | 856 | 856, 241, 42, 30 | 18 | 2360 |
| 22 | I | 0 | 899 | 898, 497, 402, 235 | 218 | 149 |
| 22 | I | 1 | 951 | 951, 118, 35, 14 | 5 | 2858 |

| Initialization | Cell | AUC | First sampled optimal step | Optimal through last sample from | Final normalized return |
|---:|:---:|---:|---:|---:|---:|
| 11 | AA | 0.09359 | not observed | not observed | 0.13167 |
| 11 | IA | 0.56272 | 1500 | not observed | 0.13167 |
| 11 | AI | 0.71473 | 1500 | 1500 | 1.00000 |
| 11 | II | 0.55923 | 2250 | 2250 | 1.00000 |
| 22 | AA | 0.40318 | 2750 | 4500 | 1.00000 |
| 22 | IA | -0.03781 | not observed | not observed | -0.03852 |
| 22 | AI | 0.87019 | 500 | 1000 | 1.00000 |
| 22 | II | 0.14360 | 4250 | 4250 | 1.00000 |

Cells name learner first, tape second: A=aged, I=injected. The JSON includes all original samples and branch counts for each 1,000-interaction block.

## Interpretation limits

Branch entries count actor experience, not replay minibatch exposure. Successful endings count observed branch rewards, not optimal policy evaluations. First crossings use the saved 250-interaction grid; staying optimal means only at every remaining sampled evaluation. Timing and coverage are associated with performance here, not identified causes. Both initializations share a structural world and anchor, but their aged weights, replay contents and post-switch action streams differ. I did not add an aggregate test or an uncertainty interval for these selected cases.
