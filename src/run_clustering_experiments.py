"""Run HDBSCAN and GMM impact clustering plus Linear Regression loss estimation.

All monetary values remain in the source dataset's reported units; verify the
currency/unit from BIPAD documentation before interpreting them financially.
"""
from pathlib import Path
import ast
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Nepal_Disaster_Data_2020_2026.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
SEED = 42
MAX_FIT = 5000
EVAL_N = 2500
GMM_COMPONENTS = [2, 3, 4, 5, 6]
HDBSCAN_CONFIGS = [(30, 10), (50, 10), (100, 10)]
IMPACT_FEATURES = [
    "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount",
    "peopleAffectedCount", "familyAffectedCount",
    "infrastructureDestroyedCount", "infrastructureAffectedHouseCount",
    "infrastructureDestroyedRoadCount", "infrastructureDestroyedBridgeCount",
    "infrastructureEconomicLoss", "agricultureEconomicLoss", "estimatedLoss",
]
REGRESSION_NUMERIC = [
    "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount",
    "peopleAffectedCount", "familyAffectedCount", "familyRelocatedCount",
    "familyEvacuatedCount", "livestockDestroyedCount",
    "infrastructureDestroyedCount", "infrastructureDestroyedHouseCount",
    "infrastructureAffectedHouseCount", "infrastructureDestroyedRoadCount",
    "infrastructureAffectedRoadCount", "infrastructureDestroyedBridgeCount",
    "infrastructureAffectedBridgeCount", "infrastructureDestroyedElectricityCount",
    "infrastructureAffectedElectricityCount", "incident_year",
]
REGRESSION_CATEGORICAL = ["hazard"]
TARGET = "estimatedLoss"


def numeric_frame(df, columns):
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def cluster_metrics(X_eval, labels, algorithm, params, fit_seconds, fit_n):
    labels = np.asarray(labels)
    non_noise = labels != -1 if algorithm == "HDBSCAN" else np.ones(len(labels), dtype=bool)
    X_m = X_eval[non_noise]
    y_m = labels[non_noise]
    unique = np.unique(y_m)
    noise_fraction = float(np.mean(labels == -1)) if algorithm == "HDBSCAN" else 0.0
    result = {
        "algorithm": algorithm, **params,
        "silhouette": np.nan, "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan, "clusters_found": int(len(unique)),
        "noise_fraction": noise_fraction, "fit_seconds": round(fit_seconds, 3),
        "fit_sample_n": int(fit_n), "evaluation_sample_n": int(len(labels)),
        "evaluation_non_noise_n": int(len(y_m)),
    }
    if len(unique) > 1 and len(unique) < len(y_m) and len(y_m) >= 3:
        try:
            result["silhouette"] = float(silhouette_score(X_m, y_m))
            result["davies_bouldin"] = float(davies_bouldin_score(X_m, y_m))
            result["calinski_harabasz"] = float(calinski_harabasz_score(X_m, y_m))
        except (ValueError, FloatingPointError):
            pass
    return result


def add_coordinates_and_hazard(df):
    out = df.copy()
    if "point" in out.columns:
        def coords(value):
            try:
                obj = json.loads(value) if isinstance(value, str) else {}
                xy = obj.get("coordinates", [])
                if len(xy) >= 2:
                    return pd.Series([float(xy[0]), float(xy[1])])
            except (ValueError, TypeError, AttributeError):
                try:
                    obj = ast.literal_eval(value)
                    xy = obj.get("coordinates", [])
                    if len(xy) >= 2:
                        return pd.Series([float(xy[0]), float(xy[1])])
                except (ValueError, SyntaxError, TypeError, AttributeError):
                    pass
            return pd.Series([np.nan, np.nan])
        xy = pd.DataFrame(out["point"].map(coords).tolist(), columns=["longitude", "latitude"], index=out.index)
        out["longitude"], out["latitude"] = xy["longitude"], xy["latitude"]
    if "title" in out.columns:
        extracted = out["title"].fillna("").astype(str).str.extract(r"^\s*(.*?)\s+at\s+(.+?)\s*$", flags=2)
        out["hazard_name"] = extracted[0].fillna("Unknown / not extracted").str.strip()
    if "incidentOn" in out.columns:
        out["incident_year"] = pd.to_datetime(out["incidentOn"], errors="coerce", utc=True).dt.year
    return out


def run_clustering(df):
    n = len(df)
    available = [c for c in IMPACT_FEATURES if c in df.columns]
    if len(available) < 3:
        raise ValueError("Dataset does not contain enough impact features for clustering.")
    raw = numeric_frame(df, available)
    loss_reported = raw[TARGET].notna().astype(float) if TARGET in raw else pd.Series(0.0, index=df.index)
    for col in raw.columns:
        raw[col] = raw[col].where(raw[col] >= 0)
        raw[col] = np.log1p(raw[col])
    raw["estimatedLoss_reported"] = loss_reported
    raw = raw.dropna(axis=1, how="all")
    raw = raw.loc[:, raw.nunique(dropna=True) > 1]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values remain after clustering preprocessing.")

    rng = np.random.default_rng(SEED)
    fit_idx = np.sort(rng.choice(n, size=min(MAX_FIT, n), replace=False))
    eval_local = np.sort(rng.choice(len(fit_idx), size=min(EVAL_N, len(fit_idx)), replace=False))
    X_fit, X_eval = X[fit_idx], X[fit_idx[eval_local]]
    results, hdb_models, gmm_models = [], {}, {}
    hdb_labels_by_cfg = {}

    for min_cluster_size, min_samples in HDBSCAN_CONFIGS:
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True)
        start = time.perf_counter()
        labels_fit = model.fit_predict(X_fit)
        elapsed = time.perf_counter() - start
        eval_labels = labels_fit[eval_local]
        params = {"min_cluster_size": min_cluster_size, "min_samples": min_samples}
        results.append(cluster_metrics(X_eval, eval_labels, "HDBSCAN", params, elapsed, len(fit_idx)))
        hdb_models[(min_cluster_size, min_samples)] = model
        hdb_labels_by_cfg[(min_cluster_size, min_samples)] = labels_fit
        print(f"HDBSCAN min_cluster_size={min_cluster_size} noise={np.mean(labels_fit == -1):.3f} time={elapsed:.2f}s")

    for k in GMM_COMPONENTS:
        model = GaussianMixture(n_components=k, covariance_type="diag", random_state=SEED, n_init=2, reg_covar=1e-5, max_iter=200)
        start = time.perf_counter()
        model.fit(X_fit)
        elapsed = time.perf_counter() - start
        eval_labels = model.predict(X_eval)
        params = {"n_components": k}
        results.append(cluster_metrics(X_eval, eval_labels, "GMM", params, elapsed, len(fit_idx)))
        gmm_models[k] = model
        print(f"GMM components={k} time={elapsed:.2f}s")

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUT / "clustering_model_comparison.csv", index=False)

    # Select a valid model using silhouette first, then DB index. If no HDBSCAN
    # configuration yields valid metrics, fall back to the best valid GMM.
    valid = comparison.dropna(subset=["silhouette"]).copy()
    if valid.empty:
        chosen_algorithm, chosen_params = "GMM", {"n_components": 3}
    else:
        best = valid.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
        chosen_algorithm = best["algorithm"]
        chosen_params = ({"min_cluster_size": int(best["min_cluster_size"]), "min_samples": int(best["min_samples"])}
                         if chosen_algorithm == "HDBSCAN" else {"n_components": int(best["n_components"])})

    # Keep both sets of labels in one row per incident so the dashboard can switch models.
    hdb_cfg = (chosen_params["min_cluster_size"], chosen_params["min_samples"]) if chosen_algorithm == "HDBSCAN" else (50, 10)
    hdb_model = hdb_models[hdb_cfg]
    hdb_full = np.full(n, -1, dtype=int)
    hdb_full[fit_idx] = hdb_labels_by_cfg[hdb_cfg]
    # GMM labels are assigned to all records; model fit remains limited to the reproducible sample.
    gmm_k = chosen_params["n_components"] if chosen_algorithm == "GMM" else 3
    gmm_model = gmm_models[gmm_k]
    gmm_full = gmm_model.predict(X)
    df_out = add_coordinates_and_hazard(df)
    df_out["hdbscan_cluster"] = hdb_full
    df_out["gmm_cluster"] = gmm_full
    # Legacy/default fields remain available for compatibility; default to GMM because it predicts all rows.
    df_out["impact_cluster"] = gmm_full
    df_out["cluster_algorithm"] = f"GMM (n_components={gmm_k})"
    df_out["cluster_assignment_scope"] = "all rows; model fit on reproducible sample"
    keep = [c for c in [
        "id_incident", "title", "incidentOn", "incident_year", "hazard", "hazard_name",
        "peopleDeathCount", "peopleMissingCount", "peopleInjuredCount", "peopleAffectedCount",
        "familyAffectedCount", "infrastructureDestroyedCount", "infrastructureAffectedHouseCount",
        "infrastructureEconomicLoss", "agricultureEconomicLoss", "estimatedLoss",
        "longitude", "latitude", "hdbscan_cluster", "gmm_cluster", "impact_cluster",
        "cluster_algorithm", "cluster_assignment_scope",
    ] if c in df_out.columns]
    df_out[keep].to_csv(OUT / "impact_cluster_assignments.csv", index=False)

    profile_rows = []
    for algorithm, col in [("HDBSCAN", "hdbscan_cluster"), ("GMM", "gmm_cluster")]:
        valid_rows = df_out[df_out[col] >= 0]
        if valid_rows.empty:
            continue
        agg_cols = {c: "median" for c in ["peopleDeathCount", "peopleInjuredCount", "peopleAffectedCount", "infrastructureDestroyedCount", "estimatedLoss"] if c in valid_rows.columns}
        grouped = valid_rows.groupby(col).agg(incidents=("id_incident", "size") if "id_incident" in valid_rows else (col, "size"), **{f"median_{k}": (k, v) for k, v in agg_cols.items()}).reset_index().rename(columns={col: "cluster"})
        grouped.insert(0, "algorithm", algorithm)
        if "hazard_name" in valid_rows:
            modes = valid_rows.groupby(col)["hazard_name"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
            grouped["most_common_hazard"] = grouped["cluster"].map(modes)
        profile_rows.append(grouped)
    profiles = pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    profiles.to_csv(OUT / "impact_cluster_profiles.csv", index=False)

    summary = {
        "dataset_rows": int(n), "source_columns": int(df.shape[1]),
        "clustering_features": list(raw.columns), "fit_sample_n": int(len(fit_idx)),
        "evaluation_sample_n": int(len(eval_local)), "selected_algorithm_by_silhouette": chosen_algorithm,
        "selected_parameters": chosen_params,
        "hdbscan_assignment_scope": "fit sample only; -1 means noise or not fitted outside the sample",
        "gmm_assignment_scope": "all rows; model fit on reproducible sample",
        "random_seed": SEED,
    }
    (OUT / "clustering_run_summary.json").write_text(json.dumps(summary, indent=2))
    return comparison


def run_loss_regression(df):
    """Train/test Linear Regression on records with known estimatedLoss.

    Loss-component fields are deliberately excluded as predictors to reduce target
    leakage. Predictions are retrospective estimates based on recorded impacts,
    not pre-event forecasts. The model predicts log1p(loss), then converts back.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from the dataset.")
    data = df.copy()
    if "incidentOn" in data.columns:
        data["incident_year"] = pd.to_datetime(data["incidentOn"], errors="coerce", utc=True).dt.year
    y_raw = pd.to_numeric(data[TARGET], errors="coerce")
    known = y_raw.notna() & y_raw.ge(0)
    model_df = data.loc[known].copy()
    y = np.log1p(y_raw.loc[known].astype(float))
    numeric_cols = [c for c in REGRESSION_NUMERIC if c in model_df.columns]
    categorical_cols = [c for c in REGRESSION_CATEGORICAL if c in model_df.columns]
    if not numeric_cols and not categorical_cols:
        raise ValueError("No suitable predictor columns were found for loss estimation.")
    X = model_df[numeric_cols + categorical_cols].copy()
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].where(X[c] >= 0)
        X[c] = np.log1p(X[c])
    for c in categorical_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    train_idx, test_idx = train_test_split(np.arange(len(model_df)), test_size=0.2, random_state=SEED)
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    start = time.perf_counter()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_seconds = time.perf_counter() - start
    pred_log = model.predict(X.iloc[test_idx])
    pred_loss = np.maximum(0, np.expm1(pred_log))
    actual_loss = y_raw.loc[known].iloc[test_idx].to_numpy(dtype=float)
    metrics = {
        "model": "Linear Regression (log1p target)",
        "target": TARGET,
        "known_target_rows": int(len(model_df)),
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "features": ";".join(numeric_cols + categorical_cols),
        "mae_reported_units": float(mean_absolute_error(actual_loss, pred_loss)),
        "rmse_reported_units": float(np.sqrt(mean_squared_error(actual_loss, pred_loss))),
        "r2_original_scale": float(r2_score(actual_loss, pred_loss)),
        "r2_log1p_scale": float(r2_score(y.iloc[test_idx], pred_log)),
        "fit_seconds": round(fit_seconds, 3),
        "target_transform": "log1p; predictions converted with expm1 and clipped at zero",
        "excluded_features_for_leakage_control": "infrastructureEconomicLoss; agricultureEconomicLoss; estimatedLoss",
        "monetary_units_note": "Source currency/unit not independently verified",
    }
    pd.DataFrame([metrics]).to_csv(OUT / "loss_regression_metrics.csv", index=False)

    # Prepare features for every incident, including those whose loss is missing.
    X_all = data.reindex(columns=numeric_cols + categorical_cols).copy()
    for c in numeric_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
        X_all[c] = X_all[c].where(X_all[c] >= 0)
        X_all[c] = np.log1p(X_all[c])
    for c in categorical_cols:
        X_all[c] = X_all[c].fillna("Unknown").astype(str)
    all_pred = np.maximum(0, np.expm1(model.predict(X_all)))
    predictions = pd.DataFrame(index=data.index)
    if "id_incident" in data.columns:
        predictions["id_incident"] = data["id_incident"]
    if "title" in data.columns:
        predictions["title"] = data["title"]
    if "incidentOn" in data.columns:
        predictions["incidentOn"] = data["incidentOn"]
    if "hazard" in data.columns:
        predictions["hazard"] = data["hazard"]
    predictions["actual_estimatedLoss"] = y_raw
    predictions["linear_regression_predicted_loss"] = all_pred
    predictions["loss_was_reported"] = y_raw.notna()
    predictions["loss_estimate_source"] = np.where(y_raw.notna(), "reported value available; prediction is a model comparison", "model estimate because reported value is missing")
    predictions.to_csv(OUT / "estimated_loss_predictions.csv", index=False)
    print("Linear Regression: MAE={:,.2f}; RMSE={:,.2f}; R2={:.4f}".format(metrics["mae_reported_units"], metrics["rmse_reported_units"], metrics["r2_original_scale"]))
    return metrics


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA, low_memory=False)
    if "id_incident" in df and df["id_incident"].duplicated().any():
        warnings.warn("Duplicate incident IDs exist; verify that the data contains one row per incident.")
    print(f"Loaded {len(df):,} rows from {DATA.name}")
    print("Running HDBSCAN and GMM clustering...")
    run_clustering(df)
    print("Training Linear Regression for estimated loss...")
    run_loss_regression(df)
    print("Wrote clustering_model_comparison.csv, impact_cluster_assignments.csv, impact_cluster_profiles.csv, loss_regression_metrics.csv, estimated_loss_predictions.csv")


if __name__ == "__main__":
    main()
