# Data-Driven Disaster Management and Impact Analysis in Nepal

## Assignment-first project guide

This repository is prepared for the **ST7072CEM Machine Learning coursework**. The assessed core is the comparison of three classification algorithms on one real-world problem using the supplied Nepal disaster incident/loss extract:

1. Decision Tree
2. Logistic Regression
3. Linear Support Vector Machine (`LinearSVC`)

All three models use the same project-defined recorded-impact target, the same stratified 80/20 train/test split, the same preprocessing pipeline, and comparable metrics. The primary experiment is `notebooks/02_machine_learning_experiments.ipynb`.

A standalone reproducibility runner is also provided at `scripts/run_paper_experiments.py`. It runs the exact three-model experiment described in the accompanying paper and regenerates the comparison table, per-class scores, confusion-matrix counts/figures, class distribution, software/run metadata, and dataset-integrity audit. This runner is the fastest reproducible route for regenerating the paper's quantitative results; the notebook remains the primary interactive coursework record.

The project proposal is broader than this assignment. Notebook 01 provides data-quality checks and exploratory data analysis. The dashboard and additional clustering/regression/NLP experiments are **supporting prototype/extension work**; they are not substitutes for the three-classifier comparison and should receive less emphasis in the assessed report.

## Dataset and integrity notes

- Data file: `data/raw/Nepal_Disaster_Data_2020_2026.csv`
- Supplied extract: **64,131 rows and 67 original columns**.
- The incident dates run from 14 April 2011 to 27 September 2026, so the filename does not describe the complete date range.
- A previous merge check reported 64,105 rows. Reconcile the 26-row difference against the source exports if exact provenance is required; do not drop rows simply to match the earlier number.
- `estimatedLoss` is missing in 32,300 records, zero in 866 and positive in 30,965. Missing is not equivalent to zero.
- Verify the monetary units/currency from source documentation before interpreting loss values.
- `hazard` and `event` should be treated as identifiers unless an authoritative lookup is joined.
- Coordinates and ward identifiers have not been independently validated against official administrative boundaries.

## Core experiment

The target is a **researcher-defined recorded-impact category**, not an official BIPAD/NDRRMA severity label:

- **High:** any recorded death or missing person; at least 5 injured; at least 100 people affected; or at least 10 infrastructure units destroyed.
- **Moderate:** at least one recorded injury, affected person/family, or destroyed infrastructure when High criteria do not apply.
- **Low:** no threshold is met among the available target indicators.

Records with all target-defining indicators missing remain unlabelled and are excluded. For partially reported records, Low means low *recorded* impact under this rule, not confirmed absence of impact. Target-defining impact fields are excluded from predictor features to reduce direct leakage.

The notebook reports accuracy, macro precision, macro recall, macro-F1, class-wise classification metrics, and confusion matrices. Select the preferred baseline using macro-F1 and per-class recall, not accuracy alone. Results are retrospective and are not suitable for operational emergency decisions without external validation.

## Setup and execution (macOS/Linux)

Use Python 3.10 or later. From the extracted project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open the project root in Jupyter and run:

1. `notebooks/01_data_quality_and_eda.ipynb`
2. `notebooks/02_machine_learning_experiments.ipynb` (**required core experiment**)
3. `python scripts/run_paper_experiments.py` to reproduce the paper tables and figures

Other notebooks are optional extensions. The notebooks use the included CSV and do not call the BIPAD API. The core ML notebook exports `reports/impact_classification_model_comparison.csv` and `reports/supplementary_loss_regression_metrics.csv` when run. To regenerate the paper-specific results and audit artefacts in one command, run `python scripts/run_paper_experiments.py` from the project root. To run the target-rule tests, install pytest and run `python -m pytest -q tests`.

Optional dashboard:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard includes year/date, hazard, source, verification, incident-title and recorded-impact-type filters. Its separate ML Lab uses LinearSVC (not RBF SVC) for a supplementary hazard-category benchmark; that task is distinct from the paper's Low/Moderate/High recorded-impact classifier. Do not mix its metrics with the primary paper results.

## Assignment submission requirements

The ZIP is the code package, not the entire coursework submission. Submit separately or include as requested by Campus 4.0:

- An approximately **6,000-word individual report**, formatted using the IEEE conference proceedings template.
- This Python/Jupyter project ZIP.
- Clear setup and execution instructions (provided here; keep them in the submission ZIP).
- A short screencast/video of the notebook running, or a link to it.

See `ASSIGNMENT_ALIGNMENT.md` for the brief-to-project checklist and `REPORT_OUTLINE.md` for a report structure that follows the assignment's requested sections.

## Research limitations to discuss

- Labels reflect recorded outcomes and reporting completeness, not necessarily true event impact.
- A random split does not establish generalisation to later years; a chronological holdout is a useful robustness check.
- Confirm thresholds and label logic are defensible; clearly identify them as researcher-defined.
- Confirm hazard/event IDs, administrative geography and currency units from authoritative sources.
- The Linear Regression extension is weak on the current run and must not be presented as successful loss prediction without stronger evidence.
- NLP clusters and impact clusters are exploratory; inspect cluster profiles and do not assign unsupported real-world meanings.
- The work does not establish causality and is not an operational early-warning system.

## Data citation

Cite the original BIPAD Disaster Information Management System/API and the exact dataset/export access date required by your institution. The API root used during project setup was `https://bipadportal.gov.np/api/`. This README is not a substitute for citing the original data source.


## Supplementary clustering comparison
Run `python src/run_three_clustering_algorithms.py` to evaluate K-Means, DBSCAN and GMM on a reproducible sample. Results are written to `reports/three_clustering_algorithms_comparison.csv`; methodology and caveats are in `reports/THREE_CLUSTERING_RESULTS.md`. These unsupervised clusters are supplementary and do not replace the three-classifier experiment.


## Dashboard structure (coursework-aligned)
The Streamlit dashboard contains five sections: Overview, Exploratory Data Analysis, Data Quality, Incident Records, and ML Lab. Sidebar filters apply to descriptive views and incident records. ML Lab presents the fixed held-out results for the three core classifiers (Decision Tree, Logistic Regression, Linear SVM) so its reported scores remain comparable with the paper. The previous six-algorithm lab and clustering interface have been removed from the dashboard. Supplementary clustering notebooks/reports, if retained in the research archive, are not presented as part of the assessed ML Lab.

Run from the project root:
```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```


### Dashboard navigation update
The dashboard now places filters on the Overview page and includes a separate **Clustering Analysis** tab for the supplementary K-Means, DBSCAN, and GMM comparison. The **ML Lab** remains the three supervised paper classifiers (Decision Tree, Logistic Regression, Linear SVM).
