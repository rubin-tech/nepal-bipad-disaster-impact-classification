"""On-demand classification and clustering benchmarks for the Nepal disaster dashboard.

Classification target: hazard category parsed from the reported incident title.
Predictors intentionally exclude title, hazard IDs, and post-event impact/loss counts.
Therefore, this is a historical hazard-category classification experiment, not a
pre-disaster early-warning forecast.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, davies_bouldin_score,
    f1_score, precision_score, recall_score, silhouette_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42


def _prepared_classification_data(df: pd.DataFrame, min_class_rows: int = 30):
    data = df.copy()
    if "hazard_name" not in data.columns:
        raise ValueError("Could not find hazard_name. The dashboard derives it from the incident title.")

    # Keep the classification target grounded in the supplied dataset.
    target = data["hazard_name"].fillna("").astype(str).str.strip()
    valid = target.ne("") & ~target.str.lower().isin({"unknown / not extracted", "unknown", "nan", "none"})
    data, target = data.loc[valid].copy(), target.loc[valid]
    counts = target.value_counts()
    eligible = counts[counts >= min_class_rows].index
    eligible = list(eligible)
    data = data.loc[target.isin(eligible)].copy()
    target = target.loc[target.isin(eligible)]
    if len(eligible) < 2 or len(data) < 100:
        raise ValueError(
            f"Not enough labeled rows/classes after filtering. Need at least 2 hazard classes "
            f"with {min_class_rows}+ records each; found {len(eligible)} classes and {len(data)} rows."
        )

    # Avoid target leakage: no title, hazard ID, narrative, or impact/loss fields.
    candidate_features = [
        "incident_year", "incident_month", "incident_day_of_year",
        "longitude", "latitude", "source", "dataSource", "verified", "approved",
    ]
    feature_cols = [c for c in candidate_features if c in data.columns]
    if not feature_cols:
        raise ValueError("No supported classification features are available in the dataset.")
    if len(data) > 20000:
        # Stratified, reproducible cap keeps SVM runtime manageable on the full BIPAD export.
        data, _unused = train_test_split(
            data, train_size=20000, random_state=RANDOM_STATE, stratify=target
        )
        target = data["hazard_name"].astype(str)
    X = data[feature_cols].copy()
    numeric_cols = [c for c in feature_cols if c in {
        "incident_year", "incident_month", "incident_day_of_year", "longitude", "latitude"
    }]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_cols:
        X[c] = X[c].fillna("Unknown").astype(str)
    # Avoid classes too small to appear in both train and test sets.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target.astype(str))
    class_counts = pd.Series(y).value_counts()
    keep_codes = set(class_counts[class_counts >= 5].index.tolist())
    keep = pd.Series(y, index=X.index).isin(keep_codes)
    X, y = X.loc[keep], y[keep.to_numpy()]
    return X, y, feature_cols, numeric_cols, categorical_cols, label_encoder, data.loc[X.index]


def run_classification_benchmarks(df: pd.DataFrame, min_class_rows: int = 30) -> dict[str, Any]:
    """Fit and compare LinearSVC, XGBoost, and Logistic Regression.

    This dashboard benchmark predicts a hazard category from incident context;
    it is distinct from the paper experiment that classifies recorded impact.
    """
    from xgboost import XGBClassifier

    X, y, feature_cols, numeric_cols, categorical_cols, encoder, kept = _prepared_classification_data(df, min_class_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    models = {
        "Linear SVM": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "XGBoost Classifier": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", XGBClassifier(
                n_estimators=220, max_depth=5, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss",
                tree_method="hist", random_state=RANDOM_STATE,
                n_jobs=2, verbosity=0,
            )),
        ]),
    }

    rows, reports, predictions = [], {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        elapsed = time.perf_counter() - start
        rows.append({
            "algorithm": name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "fit_and_predict_seconds": elapsed,
        })
        reports[name] = classification_report(
            y_test, pred, labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_, output_dict=True, zero_division=0,
        )
        predictions[name] = {
            "y_true": y_test.copy(), "y_pred": np.asarray(pred),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=np.arange(len(encoder.classes_))),
        }
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    class_counts = kept["hazard_name"].value_counts().rename_axis("hazard_class").reset_index(name="records")
    return {
        "results": results, "reports": reports, "predictions": predictions,
        "class_names": encoder.classes_.tolist(), "class_counts": class_counts,
        "feature_columns": feature_cols, "rows_used": len(X),
        "train_rows": len(X_train), "test_rows": len(X_test),
    }


def _impact_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    columns = [
        "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount",
        "peopleAffectedCount", "familyAffectedCount", "infrastructureDestroyedCount",
        "livestockDestroyedCount", "estimatedLoss",
    ]
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 3:
        raise ValueError("At least three numeric impact features are required for clustering.")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Missing impact values remain missing until median imputation; not silently treated as zero.
    X = X.fillna(X.median()).fillna(0).clip(lower=0)
    X = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled


def run_clustering_benchmarks(df: pd.DataFrame, sample_size: int = 5000, n_clusters: int = 4) -> dict[str, Any]:
    """Compare HDBSCAN, Gaussian Mixture Model, and K-Means on impact profiles."""
    data = df.copy()
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=RANDOM_STATE)
    X_raw, X_scaled = _impact_matrix(data)
    rows, labels_by_model = [], {}

    def score_labels(name: str, labels: np.ndarray, seconds: float, noise_fraction: float = 0.0):
        labels = np.asarray(labels, dtype=int)
        labels_by_model[name] = labels
        valid = labels != -1 if name == "HDBSCAN" else np.ones(len(labels), dtype=bool)
        unique = np.unique(labels[valid])
        metrics = {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
        if len(unique) >= 2 and valid.sum() > len(unique):
            # Metric calculations on a deterministic capped subset keep UI responsive.
            ids = np.flatnonzero(valid)
            if len(ids) > 3000:
                rng = np.random.default_rng(RANDOM_STATE)
                ids = np.sort(rng.choice(ids, 3000, replace=False))
            try:
                metrics["silhouette"] = silhouette_score(X_scaled[ids], labels[ids])
                metrics["davies_bouldin"] = davies_bouldin_score(X_scaled[ids], labels[ids])
                metrics["calinski_harabasz"] = calinski_harabasz_score(X_scaled[ids], labels[ids])
            except Exception:
                pass
        rows.append({
            "algorithm": name, "records_fitted": len(labels),
            "clusters_found": len(unique), "noise_fraction": noise_fraction,
            **metrics, "fit_seconds": seconds,
        })

    start = time.perf_counter()
    hdb = HDBSCAN(min_cluster_size=30, min_samples=10, metric="euclidean", n_jobs=2, copy=True)
    hdb_labels = hdb.fit_predict(X_scaled)
    hdb_seconds = time.perf_counter() - start
    score_labels("HDBSCAN", hdb_labels, hdb_seconds, float(np.mean(hdb_labels == -1)))

    start = time.perf_counter()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=RANDOM_STATE, n_init=2, reg_covar=1e-5)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_seconds = time.perf_counter() - start
    score_labels("GMM", gmm_labels, gmm_seconds)

    start = time.perf_counter()
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    km_seconds = time.perf_counter() - start
    score_labels("K-Means", km_labels, km_seconds)

    # PCA coordinates are for visualization only; clustering is performed in the standardized feature space.
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    view = data.copy()
    view["PCA 1"] = coords[:, 0]
    view["PCA 2"] = coords[:, 1]
    for name, labels in labels_by_model.items():
        view[f"{name}_cluster"] = labels
    profile_cols = [c for c in [
        "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount", "peopleAffectedCount",
        "familyAffectedCount", "infrastructureDestroyedCount", "livestockDestroyedCount", "estimatedLoss",
    ] if c in view.columns]
    profiles = []
    for name, labels in labels_by_model.items():
        temp = view.copy()
        temp["cluster"] = labels
        grouped = temp.groupby("cluster")[profile_cols].median(numeric_only=True).reset_index()
        grouped.insert(0, "algorithm", name)
        grouped.insert(2, "records", temp.groupby("cluster").size().values)
        profiles.append(grouped)
    return {
        "results": pd.DataFrame(rows).sort_values("algorithm").reset_index(drop=True),
        "view": view, "profiles": pd.concat(profiles, ignore_index=True),
        "feature_columns": list(X_raw.columns), "rows_used": len(data),
    }


def run_linear_regression_benchmark(df: pd.DataFrame) -> dict[str, Any]:
    """Fit a baseline Linear Regression for reported estimatedLoss (retrospective only)."""
    target_col = "estimatedLoss"
    if target_col not in df.columns:
        raise ValueError("The dataset does not contain estimatedLoss.")
    numeric = [c for c in [
        "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount", "peopleAffectedCount",
        "familyAffectedCount", "infrastructureDestroyedCount", "livestockDestroyedCount",
        "incident_year", "incident_month",
    ] if c in df.columns]
    categorical = [c for c in ["hazard_name", "source"] if c in df.columns]
    data = df.copy()
    y = pd.to_numeric(data[target_col], errors="coerce")
    mask = y.notna() & np.isfinite(y) & y.ge(0)
    data, y = data.loc[mask].copy(), y.loc[mask]
    if len(data) < 100:
        raise ValueError("Not enough rows with a valid reported estimatedLoss target.")
    X = data[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        X[c] = X[c].fillna("Unknown").astype(str)
    # Transform skewed loss values, while retaining original-scale metrics.
    y_log = np.log1p(y.to_numpy())
    X_train, X_test, y_train, y_test, ylog_train, ylog_test, idx_train, idx_test = train_test_split(
        X, y.to_numpy(), y_log, data.index.to_numpy(), test_size=0.2, random_state=RANDOM_STATE
    )
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers)),
        ("model", LinearRegression()),
    ])
    start = time.perf_counter()
    model.fit(X_train, ylog_train)
    pred_log = model.predict(X_test)
    elapsed = time.perf_counter() - start
    # Bound extrapolation to the observed training-target range: plain linear models
    # can otherwise predict enormous log values for sparse one-hot/impact combinations.
    max_train_log = float(np.log1p(np.max(y_train)))
    pred_log_bounded = np.clip(pred_log, 0, max_train_log)
    pred = np.expm1(pred_log_bounded)
    metrics = pd.DataFrame([{
        "algorithm": "Linear Regression", "known_target_rows": len(data),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "mae_reported_units": mean_absolute_error(y_test, pred),
        "rmse_reported_units": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2_original_scale": r2_score(y_test, pred),
        "r2_log1p_scale": r2_score(ylog_test, pred_log_bounded), "fit_seconds": elapsed,
    }])
    pred_frame = pd.DataFrame({
        "id_incident": data.loc[idx_test, "id_incident"].to_numpy() if "id_incident" in data.columns else idx_test,
        "actual_estimatedLoss": y_test,
        "linear_regression_predicted_loss": pred,
    })
    return {"metrics": metrics, "predictions": pred_frame, "feature_columns": numeric + categorical}
