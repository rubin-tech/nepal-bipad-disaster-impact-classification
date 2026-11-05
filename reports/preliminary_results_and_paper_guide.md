# Reproduced results and paper-writing guide

The core paper experiment was independently rerun using `python scripts/run_paper_experiments.py` against the CSV included in this project. The script writes the result tables, confusion matrices, figures, software metadata and data-integrity audit in this folder. Results below are from the verified run on 3 November 2026.

## Dataset and target

- Current supplied extract: **64,131 rows × 67 original fields**.
- Parsed incident dates in Nepal local time: 14 April 2011 through 27 September 2026.
- The filename contains `2020_2026`, but the actual incident dates extend back to 2011.
- An earlier merge check reported 64,105 rows, 26 fewer than the current file. This remains an unresolved source-version discrepancy; no rows were removed to force agreement.
- `estimatedLoss`: 32,300 missing, 866 reported zero, and 30,965 positive. Missing is not equivalent to zero; currency/units are not confirmed.
- The classification target is a researcher-defined Low/Moderate/High recorded-impact category, not an official BIPAD/NDRRMA severity label.

Target rule: High if any recorded death or missing person, at least five injuries, at least 100 affected people, or at least ten destroyed infrastructure units; Moderate if any injury, affected person/family, or destroyed infrastructure is recorded but no High threshold is met; Low if none of these thresholds is met among available indicators. Records with every target-defining field missing remain unlabelled. For partially reported rows, Low means low recorded impact under the available fields, not verified absence of impact.

| Target class | Records | Share |
|---|---:|---:|
| Moderate | 47,094 | 73.43% |
| Low | 9,935 | 15.49% |
| High | 7,102 | 11.07% |

## Core classification experiment

All three models use the same contextual features (hazard ID, date components, coordinates, verification/approval and source fields), preprocessing pipeline, stratified 80/20 split and random state 42. The held-out test set contains 12,827 records; training contains 51,304. Target-defining impact fields are not included as predictors.

| Algorithm | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Linear SVM (`LinearSVC`) | 0.7628 | 0.6160 | 0.6141 | 0.6149 |
| Decision Tree | 0.6671 | 0.5767 | 0.7078 | 0.5978 |
| Logistic Regression | 0.6954 | 0.5681 | 0.6574 | 0.5929 |

**Important correction for manuscript wording:** 0.7628 (76.28%) is the Linear SVM's **accuracy**, not its precision. Its macro precision is 0.6160 (61.60%). The manuscript abstract/table must not call 76.28% precision.

### Per-class F1 (held-out test set)

| Algorithm | Low F1 | Moderate F1 | High F1 |
|---|---:|---:|---:|
| Decision Tree | 0.6006 | 0.7543 | 0.4385 |
| Logistic Regression | 0.5272 | 0.7976 | 0.4540 |
| Linear SVM | 0.5004 | 0.8589 | 0.4853 |

The test supports are Low=1,987, Moderate=9,419 and High=1,421. Linear SVM has the strongest macro-F1 and accuracy, but its Low and High class F1 scores remain modest. Decision Tree has higher Low recall (0.7811) and High recall (0.7051) than the other models, at the cost of lower overall accuracy and macro-F1. The choice of preferred model therefore depends on the cost of missing each impact category; macro-F1 alone is not sufficient for an operational decision.

## Supplementary regression

The existing supplementary Linear Regression experiment predicts reported `estimatedLoss` using a log1p target and compares against a simple median baseline. Its earlier recorded run produced MAE about 2.279 million reported units, RMSE about 59.647 million and original-scale R² about -0.0014. It did not demonstrate useful predictive value. Do not describe it as a successful loss-prediction model, and do not attach a currency symbol until the units are verified.

## Interpretation and limitations

This is retrospective classification of already-recorded impact categories. It is not a pre-event forecast, hazard prediction, validated severity scale, or emergency-alert system. Missing impact fields can represent incomplete reporting, not zero impact. The random split does not test generalisation to future years or unseen events. Hazard/event identifiers are not human-readable hazard names until verified lookup tables are joined. Coordinates and administrative areas require validation against authoritative boundaries. Results show an exploratory signal in event-context attributes, not operational readiness.

## Reproduction

```bash
python -m pip install -r requirements.txt
python scripts/run_paper_experiments.py
python -m pytest -q tests
```

See `paper_experiment_metadata.json`, `dataset_integrity_audit.json`, `impact_classification_per_class.csv`, `impact_classification_confusion_matrices.csv` and `figures/` for the detailed evidence.
