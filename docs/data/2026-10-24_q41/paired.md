## bfcl_multi_turn (reference ud, margin 3.0 points, process gate +0.5 points)

| File | tasks | K | pass@1 | E[p(1-p)] | self-discordance | paired tasks (samples) | Δ pass@1 | SE (clustered) | 90 % | 95 % | bootstrap 90 % | verdict | Δ malformed+runaway | upper 90 % | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ud | 358 | 1 | 31.01 | nan | nan % | — | — | — | — | — | — | — | — | — | — |
| g3 | 358 | 1 | 34.08 | nan | nan % | 358 (358) | +3.07 | 2.69 (2.83) | -1.59 to +7.73 | -2.48 to +8.63 | -1.40 to +7.82 | pass | +1.40 | +4.81 | FAIL |

## bfcl_single_turn (reference ud, margin 1.5 points, process gate +0.5 points)

| File | tasks | K | pass@1 | E[p(1-p)] | self-discordance | paired tasks (samples) | Δ pass@1 | SE (clustered) | 90 % | 95 % | bootstrap 90 % | verdict | Δ malformed+runaway | upper 90 % | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ud | 3477 | 2 | 76.30 | 0.076 | 15.1 % | — | — | — | — | — | — | — | — | — | — |
| g3 | 3477 | 2 | 76.65 | 0.073 | 14.6 % | 3477 (6954) | +0.35 | 0.45 (0.44) | -0.40 to +1.09 | -0.54 to +1.23 | -0.39 to +1.08 | pass | -0.06 | +0.21 | pass |
| k2h | 3477 | 1 | 76.68 | nan | nan % | 3477 (3477) | +0.14 | 0.62 (0.61) | -0.87 to +1.16 | -1.06 to +1.35 | -0.86 to +1.15 | pass | -0.26 | +0.02 | pass |

