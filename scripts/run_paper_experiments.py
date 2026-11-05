#!/usr/bin/env python3
"""Reproduce the three-class BIPAD recorded-impact experiments reported in the paper.

Run from the project root:
    python scripts/run_paper_experiments.py

The target is researcher-defined from reported impact fields. It is not an
official BIPAD/NDRRMA severity label and is not a prospective warning target.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_utils import load_dataset, add_local_date_features, add_coordinates, impact_class

SEED = 42
LABELS = ["Low", "Moderate", "High"]
CAT_COLS = ["hazard", "verified", "approved", "source", "dataSource"]
NUM_COLS = ["incident_year", "incident_month", "incident_day_of_year", "longitude", "latitude"]
FEATURES = CAT_COLS + NUM_COLS


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUM_COLS),
    ])


def main() -> None:
    reports = ROOT / "reports"
    figures = reports / "figures"
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    started = time.perf_counter()

    raw = load_dataset()
    df = add_coordinates(add_local_date_features(raw))
    df["recorded_impact_class"] = df.apply(impact_class, axis=1)
    class_counts = (df["recorded_impact_class"].value_counts(dropna=False)
                    .rename_axis("recorded_impact_class").reset_index(name="records"))
    class_counts["proportion"] = class_counts["records"] / len(df)
    class_counts.to_csv(reports / "impact_class_distribution.csv", index=False)

    model_df = df[FEATURES + ["recorded_impact_class"]].dropna(subset=["recorded_impact_class"]).copy()
    X, y = model_df[FEATURES], model_df["recorded_impact_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=SEED
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight="balanced", random_state=SEED, max_iter=5000
        ),
    }

    comparison, per_class, confusion_rows = [], [], []
    for name, estimator in models.items():
        tick = time.perf_counter()
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        predicted = pipe.predict(X_test)
        elapsed = time.perf_counter() - tick
        comparison.append({
            "algorithm": name,
            "accuracy": accuracy_score(y_test, predicted),
            "macro_precision": precision_score(y_test, predicted, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, predicted, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
            "fit_predict_seconds": elapsed,
        })
        report = classification_report(
            y_test, predicted, labels=LABELS, output_dict=True, zero_division=0
        )
        for label in LABELS:
            per_class.append({"algorithm": name, "class": label, **report[label]})
        cm = confusion_matrix(y_test, predicted, labels=LABELS)
        for i, actual in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "algorithm": name, "actual_class": actual,
                    "predicted_class": predicted_label, "count": int(cm[i, j]),
                })
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
            ax=ax, values_format="d", colorbar=False
        )
        ax.set_title(f"{name}: recorded-impact classification")
        fig.tight_layout()
        fig.savefig(figures / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=180)
        plt.close(fig)
        print(f"{name}: accuracy={accuracy_score(y_test, predicted):.4f}; "
              f"macro-F1={f1_score(y_test, predicted, average='macro'):.4f}; "
              f"elapsed={elapsed:.2f}s")

    result_df = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    result_df.to_csv(reports / "impact_classification_model_comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(reports / "impact_classification_per_class.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(reports / "impact_classification_confusion_matrices.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Low", "Moderate", "High"]
    counts = df["recorded_impact_class"].value_counts().reindex(order, fill_value=0)
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Project-defined recorded-impact class distribution")
    ax.set_xlabel("Recorded-impact class")
    ax.set_ylabel("Incident records")
    fig.tight_layout()
    fig.savefig(figures / "impact_class_distribution.png", dpi=180)
    plt.close(fig)

    metadata = {
        "dataset_file": "data/raw/Nepal_Disaster_Data_2020_2026.csv",
        "dataset_rows": int(len(raw)), "dataset_columns": int(len(raw.columns)),
        "labelled_rows": int(len(model_df)), "unlabelled_rows": int(len(raw) - len(model_df)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "test_fraction": 0.20, "random_state": SEED, "stratified": True,
        "features": FEATURES,
        "target": "recorded_impact_class (researcher-defined; not an official severity label)",
        "class_counts": {str(k): int(v) for k, v in df["recorded_impact_class"].value_counts().items()},
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "scikit_learn_version": sklearn.__version__,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (reports / "paper_experiment_metadata.json").write_text(json.dumps(metadata, indent=2))

    audit = {
        "current_extract_rows": int(len(raw)), "current_extract_columns": int(len(raw.columns)),
        "earlier_merge_check_rows": 64105,
        "difference_rows": int(len(raw) - 64105),
        "interpretation": "Unresolved provenance/version discrepancy; no rows were removed to force a match.",
        "incident_id_column": "id_incident" if "id_incident" in raw else None,
        "incident_id_duplicate_count": int(raw["id_incident"].duplicated().sum()) if "id_incident" in raw else None,
        "loss_id_duplicate_count": int(raw["id_loss"].duplicated().sum()) if "id_loss" in raw else None,
        "incident_date_min_utc": str(pd.to_datetime(raw["incidentOn"], errors="coerce", utc=True).min()),
        "incident_date_max_utc": str(pd.to_datetime(raw["incidentOn"], errors="coerce", utc=True).max()),
        "estimatedLoss_missing": int(raw["estimatedLoss"].isna().sum()),
        "estimatedLoss_zero": int((pd.to_numeric(raw["estimatedLoss"], errors="coerce") == 0).sum()),
        "estimatedLoss_positive": int((pd.to_numeric(raw["estimatedLoss"], errors="coerce") > 0).sum()),
        "warning": "Do not interpret missing estimatedLoss as zero; verify currency/units from source documentation.",
    }
    (reports / "dataset_integrity_audit.json").write_text(json.dumps(audit, indent=2))

    print("\nClass counts:\n", class_counts.to_string(index=False))
    print("\nComparison:\n", result_df.to_string(index=False))
    print(f"\nOutputs saved to {reports}; total runtime {metadata['runtime_seconds']} seconds.")


if __name__ == "__main__":
    main()
