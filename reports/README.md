# Reports folder guide

Run `python scripts/run_paper_experiments.py` from the project root to regenerate the primary paper outputs from the included CSV.

## Core paper outputs

- `impact_classification_model_comparison.csv` — accuracy, macro precision, macro recall, macro-F1 and runtime for Decision Tree, Logistic Regression and Linear SVM.
- `impact_classification_per_class.csv` — class-wise precision, recall, F1 and support for each classifier.
- `impact_classification_confusion_matrices.csv` — confusion-matrix counts for each classifier.
- `impact_class_distribution.csv` — Low/Moderate/High record counts and proportions.
- `paper_experiment_metadata.json` — dataset shape, labelled/train/test sizes, features, seed and software versions.
- `dataset_integrity_audit.json` — current extract size, date range, unresolved 64,131 vs 64,105 row discrepancy and loss-field missingness audit.
- `figures/` — class distribution and three confusion-matrix figures.

## Supporting outputs

- `supplementary_loss_regression_metrics.csv` — optional secondary Linear Regression results.
- The remaining clustering/NLP CSV and JSON files support optional prototype/dashboard features. They are not the primary evidence for the three-classifier comparison. Rerun the corresponding notebook or script before citing any supplementary output in the report.

Never infer missing values are zero; verify source currency/units; and describe project-defined impact classes as researcher-defined rather than official BIPAD labels. The models retrospectively classify recorded impact categories and are not pre-event forecasts or emergency alerts.
