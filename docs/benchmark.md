# xG benchmark

**My model closes 76% of the log-loss gap** between a distance+angle baseline and StatsBomb's production model, on competitions neither model was trained on here.

Held out: **43:106, 72:107** (3,038 shots). Trained on 7,820 shots across 11 other competition/seasons: 11:27, 11:90, 2:27, 44:107, 53:106, 53:315, 55:282, 55:43, 7:108, 7:235, 9:281.

Split by competition/season, never by shot — shots inside a match share a game state and a shooter, so a random shot split leaks between train and test and inflates every number below. Penalties are excluded (fixed geometry, ~78% conversion, and they flatter AUC badly). `statsbomb_xg` is a comparison column only and is never a feature.

Data source: StatsBomb.

| model | log-loss | Brier | ROC-AUC | expected goals | observed | O/E | gap closed |
|---|--:|--:|--:|--:|--:|--:|--:|
| distance+angle | 0.2811 | 0.0786 | 0.7330 | 322.4 | 288 | 0.893 | 0% |
| mine (+context) | 0.2581 | 0.0727 | 0.7933 | 297.8 | 288 | 0.967 | **76%** |
| StatsBomb (reference) | 0.2507 | 0.0693 | 0.8035 | 282.8 | 288 | 1.018 | 100% |

![calibration](calibration.png)

## By competition

| competition/season | shots | goals | baseline | mine | StatsBomb |
|---|--:|--:|--:|--:|--:|
| 43:106 | 1,430 | 152 | 0.2972 | 0.2728 | 0.2665 |
| 72:107 | 1,608 | 136 | 0.2669 | 0.2450 | 0.2367 |

## The model that is served is the model that was measured

Training writes a pickle: a `CalibratedClassifierCV` wrapping three
`LGBMClassifier` objects. That is the right artefact for an experiment and the
wrong one to deploy — restoring it needs scikit-learn and pandas close enough to
the versions that wrote it, and nothing tells you when they are not.

So `models/export_xg_portable.py` re-serialises the model into what it actually
is:

```
p(goal) = mean over the 3 members of  1 / (1 + exp(a·margin + b))
```

Three gradient-boosted ensembles in LightGBM's own documented text format, a
two-parameter sigmoid on each, and a lookup table for the categoricals. 599 KB
gzipped, committed to the repo, and readable by any lightgbm ≥ 4. The hosted API
loads it in place of the pickle and needs neither scikit-learn nor pandas to
score a shot.

The translation is hand-written, so it is verified rather than trusted.
`tests/test_xg_portable.py` scores every non-penalty shot in the corpus both
ways and requires **exact** equality — not a tolerance, since it is the same
arithmetic on the same floats.

That test earns its place. scikit-learn calibrates whatever `decision_function`
returns, which for lightgbm is the raw margin, not a probability. Calibrating
the probability instead — the obvious reading — produces values pinned between
0.43 and 0.62. Every one of them looks like a plausible xG on a page, the
service returns 200, and the model is wrong on every shot. A smoke test passes;
an equality test over 10,858 shots does not.

`models/train_xg.py` writes both artefacts in the same run, so the deployed
model cannot drift from the one this page reports.
