from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_dataset, add_local_date_features, add_coordinates, impact_class

st.set_page_config(page_title="Nepal BIPAD | Disaster Impact Explorer", page_icon="🇳🇵", layout="wide")
LABELS = ["Low", "Moderate", "High"]
MODEL_ORDER = ["Decision Tree", "Logistic Regression", "Linear SVM"]
IMPACT_COLS = ["peopleDeathCount", "peopleMissingCount", "peopleInjuredCount", "peopleAffectedCount", "familyAffectedCount", "infrastructureDestroyedCount"]

@st.cache_data(show_spinner="Loading BIPAD incident records…")
def load_prepared_data():
    data = add_coordinates(add_local_date_features(load_dataset()))
    data["recorded_impact_class"] = data.apply(impact_class, axis=1)
    for c in IMPACT_COLS + ["estimatedLoss", "longitude", "latitude"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

@st.cache_data
def load_report_csv(filename):
    p = ROOT / "reports" / filename
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

try:
    df = load_prepared_data()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.info("Check data/raw/Nepal_Disaster_Data_2020_2026.csv and install requirements.txt.")
    st.stop()

st.title("🇳🇵 Nepal Disaster Impact Explorer")
st.caption("Retrospective analysis of recorded disaster impacts in the supplied BIPAD-derived incident dataset. Low / Moderate / High are researcher-defined recorded-impact categories, not official severity ratings.")

# Overview-page filters are applied to Overview, EDA, Data Quality and Incident Records.
# The classifier and clustering experiments remain fixed, reproducible paper results.
filtered = df.copy()

def options_for(frame, col):
    if col not in frame.columns: return []
    vals = frame[col].dropna().astype(str).str.strip()
    return sorted(v for v in vals.unique() if v and v.lower() not in {"nan", "none"})

# Source hazard values are numeric IDs. Build a readable label from incident titles
# (e.g. "Fire at ..." -> "Fire") while retaining the original IDs for filtering.
def hazard_name_lookup(frame):
    if "hazard" not in frame.columns:
        return {}
    if "title" not in frame.columns:
        return {str(v): f"Unspecified hazard (ID {v})" for v in frame["hazard"].dropna().unique()}
    import re
    names = frame[["hazard", "title"]].dropna().copy()
    names["hazard"] = names["hazard"].astype(str)
    names["Hazard name"] = names["title"].astype(str).str.extract(r"^\s*(.+?)\s+at\s+.+$", flags=re.IGNORECASE)[0].str.strip()
    names = names[names["Hazard name"].notna() & names["Hazard name"].ne("")]
    if names.empty:
        return {str(v): f"Unspecified hazard (ID {v})" for v in frame["hazard"].dropna().unique()}
    return names.groupby("hazard")["Hazard name"].agg(lambda x: x.value_counts().index[0]).to_dict()

hazard_name_by_id = hazard_name_lookup(df)
def hazard_label(value):
    key = str(value)
    return hazard_name_by_id.get(key, "Unspecified hazard")

# Shared helpers
def metric_sum(frame, col):
    return pd.to_numeric(frame[col], errors="coerce").sum(min_count=1) if col in frame else np.nan

def quality_table(frame):
    rows=[]
    for c in frame.columns:
        rows.append({"Field":c,"Data type":str(frame[c].dtype),"Missing":int(frame[c].isna().sum()),"Missing (%)":round(frame[c].isna().mean()*100,2),"Unique values":int(frame[c].nunique(dropna=True) )})
    return pd.DataFrame(rows).sort_values(["Missing (%)","Field"], ascending=[False,True])

# Navigation
(tab_overview, tab_eda, tab_quality, tab_records, tab_geo, tab_ml, tab_clustering) = st.tabs(["Overview", "Exploratory Data Analysis", "Data Quality", "Incident Records", "Geographic view", "ML Lab", "Clustering Analysis"])

with tab_overview:
    st.subheader("Overview filters")
    st.caption("Choose filters here. The same selection is used by Overview, EDA, Data Quality and Incident Records.")
    f1, f2, f3 = st.columns(3)
    with f1:
        year_options = sorted(pd.to_numeric(df["incident_year"], errors="coerce").dropna().astype(int).unique().tolist())
        year_range = st.select_slider("Incident year", options=year_options, value=(year_options[0], year_options[-1]), key="overview_year_filter")
    filtered = df[pd.to_numeric(df["incident_year"], errors="coerce").between(*year_range)].copy()
    with f2:
        hazard_ids = options_for(filtered, "hazard")
        hazard_label_to_ids = {f"{hazard_label(v)} (ID {v})": v for v in hazard_ids}
        selected_hazards = st.multiselect("Hazard ID / name", list(hazard_label_to_ids), default=[], key="overview_hazard_filter")
    if selected_hazards:
        selected_ids = [hazard_label_to_ids[label] for label in selected_hazards]
        filtered = filtered[filtered["hazard"].astype(str).isin(selected_ids)]
    with f3:
        source_options = options_for(filtered, "source")
        selected_sources = st.multiselect("Source", source_options, default=[], key="overview_source_filter")
    if selected_sources:
        filtered = filtered[filtered["source"].astype(str).isin(selected_sources)]
    f4, f5 = st.columns(2)
    with f4:
        verified_choice = st.selectbox("Verification status", ["All", "Verified", "Not verified"], key="overview_verified_filter")
    if verified_choice != "All":
        verified = filtered.get("verified", pd.Series(index=filtered.index, dtype="object")).astype("string").str.strip().str.lower()
        if verified_choice == "Verified":
            filtered = filtered[verified.isin(["true", "1", "yes"])]
        else:
            filtered = filtered[verified.isin(["false", "0", "no"])]
    with f5:
        impact_choice = st.multiselect("Recorded-impact class", LABELS, default=[], key="overview_impact_filter")
    if impact_choice:
        filtered = filtered[filtered["recorded_impact_class"].isin(impact_choice)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} incident records. ML Lab and Clustering Analysis use fixed paper experiment outputs and are not recalculated by these filters.")
    st.divider()
    if filtered.empty:
        st.warning("No records match the selected filters. Adjust the Overview filters above.")
    else:
        n=len(filtered); deaths=metric_sum(filtered,"peopleDeathCount"); injuries=metric_sum(filtered,"peopleInjuredCount"); affected=metric_sum(filtered,"peopleAffectedCount")
        k1,k2,k3,k4=st.columns(4)
        k1.metric("Incident records",f"{n:,}"); k2.metric("Recorded deaths","—" if pd.isna(deaths) else f"{deaths:,.0f}"); k3.metric("Recorded injuries","—" if pd.isna(injuries) else f"{injuries:,.0f}"); #k4.metric("People affected","—" if pd.isna(affected) else f"{affected:,.0f}")
        st.divider(); left,right=st.columns(2)
        with left:
            st.subheader("Recorded incidents by year")
            yearly=filtered.dropna(subset=["incident_year"]).groupby("incident_year").size().reset_index(name="Records")
            if not yearly.empty: st.plotly_chart(px.line(yearly,x="incident_year",y="Records",markers=True,labels={"incident_year":"Year"}), use_container_width=True, key="plotly_chart_1")
        with right:
            st.subheader("Recorded-impact class distribution")
            counts=filtered["recorded_impact_class"].value_counts().reindex(LABELS,fill_value=0).rename_axis("Class").reset_index(name="Records")
            st.plotly_chart(px.bar(counts,x="Class",y="Records",category_orders={"Class":LABELS},color="Class",title="Project-defined recorded-impact classes"),use_container_width=True, key="plotly_chart_2")
        st.subheader("Records by hazard ID")
        if "hazard" in filtered:
            hc=filtered["hazard"].fillna("Not recorded").astype(str).map(lambda v: f"{hazard_label(v)} (ID {v})").value_counts().head(12).rename_axis("Hazard").reset_index(name="Records")
            st.plotly_chart(px.bar(hc.sort_values("Records"),x="Records",y="Hazard",orientation="h"),use_container_width=True, key="plotly_chart_3")
        st.caption("Missing impact values are not assumed to mean zero. The target classes describe recorded outcomes retrospectively; they are not forecasts or official severity labels.")

with tab_eda:
    st.subheader("Exploratory data analysis")
    if filtered.empty: st.warning("No records match the current filters.")
    else:
        left,right=st.columns(2)
        with left:
            st.markdown("**Temporal pattern — annual records**")
            yearly=filtered.dropna(subset=["incident_year"]).groupby("incident_year").size().reset_index(name="Records")
            if not yearly.empty: st.plotly_chart(px.line(yearly,x="incident_year",y="Records",markers=True),use_container_width=True, key="plotly_chart_4")
        with right:
            st.markdown("**Monthly distribution**")
            months=pd.to_numeric(filtered.get("incident_month"),errors="coerce")
            m=months[months.between(1,12)].astype(int).value_counts().reindex(range(1,13),fill_value=0).rename_axis("Month").reset_index(name="Records")
            m["Month name"]=pd.to_datetime(m["Month"],format="%m").dt.month_name()
            st.plotly_chart(px.bar(m,x="Month name",y="Records"),use_container_width=True, key="plotly_chart_5")
        left,right=st.columns(2)
        with left:
            st.markdown("**Hazard frequency**")
            if "hazard" in filtered:
                h=filtered["hazard"].fillna("Not recorded").astype(str).map(lambda v: f"{hazard_label(v)} (ID {v})").value_counts().head(15).rename_axis("Hazard").reset_index(name="Records")
                st.plotly_chart(px.bar(h.sort_values("Records"),x="Records",y="Hazard",orientation="h"),use_container_width=True, key="plotly_chart_6")
        with right:
            st.markdown("**Impact class distribution**")
            cc=filtered["recorded_impact_class"].value_counts().reindex(LABELS,fill_value=0).rename_axis("Class").reset_index(name="Records")
            st.plotly_chart(px.bar(cc,x="Class",y="Records",category_orders={"Class":LABELS},color="Class"),use_container_width=True, key="plotly_chart_7")
        st.markdown("**Recorded human and infrastructure impacts**")
        avail=[c for c in IMPACT_COLS if c in filtered]
        if avail:
            summary=pd.DataFrame({"Impact field":c,"Recorded total":pd.to_numeric(filtered[c],errors="coerce").sum(min_count=1),"Non-missing records":int(filtered[c].notna().sum())} for c in avail)
            st.dataframe(summary,use_container_width=True,hide_index=True)
        st.markdown("**Reported estimated-loss distribution**")
        loss=pd.to_numeric(filtered.get("estimatedLoss"),errors="coerce").dropna()
        if loss.empty: st.info("No estimated-loss values are reported for the current filter selection.")
        else:
            st.plotly_chart(px.histogram(pd.DataFrame({"Reported estimated loss":loss}),x="Reported estimated loss",nbins=40),use_container_width=True, key="plotly_chart_8")
            st.caption(f"{filtered['estimatedLoss'].isna().sum():,} filtered records have missing estimatedLoss. Missing values are excluded, not converted to zero. Confirm currency and units before interpreting totals.")
        st.markdown("**Geographic distribution of records with valid coordinates**")
        if {"longitude","latitude"}.issubset(filtered.columns):
            geo=filtered.dropna(subset=["longitude","latitude"]).copy()
            geo=geo[geo.longitude.between(80,89)&geo.latitude.between(25,31)]
            if geo.empty: st.info("No valid Nepal-range coordinates are available for this selection.")
            else:
                if len(geo)>6000: geo=geo.sample(6000,random_state=42)
                color="recorded_impact_class" if "recorded_impact_class" in geo else None
                fig=px.scatter_geo(geo,lat="latitude",lon="longitude",color=color,hover_name="title" if "title" in geo else None,scope="asia",title="Geocoded incident records (sample capped at 6,000)")
                fig.update_geos(lataxis_range=[25,31],lonaxis_range=[80,89],showcountries=True)
                st.plotly_chart(fig,use_container_width=True, key="plotly_chart_9")
                st.caption("Map points show reported records with coordinates, not the true spatial incidence of all disasters. Coordinates have not been independently verified.")


with tab_geo:
    st.subheader("Geocoded incident records")
    st.caption("Explore the geographic distribution of reported Nepal disaster incidents. Each colour represents a recorded hazard type.")
    geo_source = filtered.copy()
    if not {"longitude", "latitude"}.issubset(geo_source.columns):
        st.warning("Longitude and latitude fields are not available in this dataset.")
    else:
        geo_source["longitude"] = pd.to_numeric(geo_source["longitude"], errors="coerce")
        geo_source["latitude"] = pd.to_numeric(geo_source["latitude"], errors="coerce")
        geo = geo_source.dropna(subset=["longitude", "latitude"]).copy()
        geo = geo[geo["longitude"].between(80, 89) & geo["latitude"].between(25, 31)]
        # Prefer readable hazard names extracted from incident titles. The raw
        # `hazard` field contains numeric source IDs, so do not expose it as a
        # user-facing label in the geographic view.
        if "hazard_name" in geo.columns:
            hazard_names = geo["hazard_name"].astype("string").str.strip()
        else:
            hazard_names = pd.Series(pd.NA, index=geo.index, dtype="string")
        if "title" in geo.columns:
            from_title = geo["title"].fillna("").astype(str).str.extract(
                r"^\s*(?P<name>.+?)\s+at\s+.+$", flags=__import__("re").IGNORECASE
            )["name"].astype("string").str.strip()
            hazard_names = hazard_names.fillna(from_title)
        geo["Hazard type"] = hazard_names.fillna("Unspecified hazard").replace("", "Unspecified hazard")
        m1, m2, m3 = st.columns(3)
        m1.metric("Geocoded records", f"{len(geo):,}")
        m2.metric("Different hazard types", f"{geo['Hazard type'].nunique():,}")
        m3.metric("Records without coordinates", f"{geo_source[['longitude','latitude']].isna().any(axis=1).sum():,}")
        hazards = sorted(geo["Hazard type"].unique().tolist())
        chosen = st.multiselect("Show hazard types on map", hazards, default=hazards, key="geo_hazard_types")
        shown = geo[geo["Hazard type"].isin(chosen)]
        st.info("The map displays a reproducible sample of up to 6,000 records for performance. Summary metrics use all geocoded records matching the Overview filters.")
        if shown.empty:
            st.info("No geocoded records match the selected hazard types.")
        else:
            if len(shown) > 6000:
                shown = shown.sample(6000, random_state=42)
            hover_cols = [c for c in ["incidentOn", "Hazard type", "recorded_impact_class"] if c in shown.columns]
            fig = px.scatter_geo(
                shown, lat="latitude", lon="longitude", color="Hazard type",
                hover_name="title" if "title" in shown.columns else None,
                hover_data=hover_cols, scope="asia",
                title="Geographic distribution of reported disaster incidents in Nepal",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_geos(
                lataxis_range=[25, 31], lonaxis_range=[80, 89],
                showcountries=True, showland=True, landcolor="#f0f0f0",
                showocean=True, oceancolor="#ffffff"
            )
            fig.update_layout(legend_title_text="Hazard type", margin=dict(l=0,r=0,t=45,b=0))
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_10")
            st.caption("Map points represent reported records with coordinates, not the true spatial incidence of all disasters. Coordinates have not been independently verified. The map is descriptive and does not replace the coursework's classification and clustering evaluation.")

with tab_quality:
    st.subheader("Data quality assessment")
    st.caption("Quality summaries below use the records remaining after the Overview filters. Missing values are distinguished from recorded zeros.")
    n=len(filtered)
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Rows in selection",f"{n:,}")
    q2.metric("Duplicate incident IDs",f"{filtered['id_incident'].duplicated().sum():,}" if "id_incident" in filtered else "N/A")
    q3.metric("Missing incident dates",f"{pd.to_datetime(filtered.get('incidentOn'),errors='coerce').isna().sum():,}" if "incidentOn" in filtered else "N/A")
    q4.metric("Missing coordinates",f"{filtered[['longitude','latitude']].isna().any(axis=1).sum():,}" if {"longitude","latitude"}.issubset(filtered.columns) else "N/A")
    st.markdown("**Field-level completeness and cardinality**")
    qt=quality_table(filtered)
    st.dataframe(qt,use_container_width=True,hide_index=True)
    st.download_button("Download data-quality profile (CSV)",qt.to_csv(index=False).encode("utf-8-sig"),"nepal_bipad_data_quality_profile.csv","text/csv")
    st.markdown("**Coordinate validation**")
    if {"longitude","latitude"}.issubset(filtered.columns):
        lon=pd.to_numeric(filtered.longitude,errors="coerce"); lat=pd.to_numeric(filtered.latitude,errors="coerce")
        valid=lon.between(-180,180)&lat.between(-90,90)
        nepal=lon.between(80,89)&lat.between(25,31)
        a,b,c=st.columns(3); a.metric("Globally valid coordinate pairs",f"{int(valid.sum()):,}"); b.metric("Nepal-range coordinate pairs",f"{int(nepal.sum()):,}"); c.metric("Invalid/out-of-range pairs",f"{int((~valid & lon.notna() & lat.notna()).sum()):,}")
    st.markdown("**Impact-field missingness**")
    impact_quality=pd.DataFrame([{"Impact field":c,"Missing records":int(filtered[c].isna().sum()),"Completeness (%)":round(filtered[c].notna().mean()*100,2)} for c in IMPACT_COLS if c in filtered])
    if not impact_quality.empty: st.dataframe(impact_quality,use_container_width=True,hide_index=True)
    st.info("Data-quality checks identify completeness and basic consistency only. They do not establish that source reports, coordinates, dates or impact values are factually correct. A missing impact value must not automatically be interpreted as zero.")

with tab_records:
    st.subheader("Filtered incident records")
    st.write(f"{len(filtered):,} records match the current Overview filters.")
    cols=["id_incident","title","incidentOn","incident_year","hazard","source","verified","recorded_impact_class"]+IMPACT_COLS+["estimatedLoss","longitude","latitude"]
    cols=[c for c in cols if c in filtered.columns]
    records_display = filtered[cols].copy()
    if "hazard" in records_display.columns:
        records_display["hazard"] = records_display["hazard"].fillna("Not recorded").astype(str).map(lambda v: f"{hazard_label(v)} (ID {v})")
        records_display = records_display.rename(columns={"hazard": "Hazard type (ID)"})
    st.dataframe(records_display,use_container_width=True,hide_index=True)
    st.download_button("Download filtered records (CSV)",records_display.to_csv(index=False).encode("utf-8-sig"),"nepal_bipad_filtered_incident_records.csv","text/csv")

with tab_ml:
    st.subheader("Three-classifier comparison required for the coursework")
    st.caption("Fixed held-out evaluation from the paper: stratified 80:20 split, random_state=42, and 12,827 test records. These results are intentionally independent of the descriptive Overview filters.")
    comparison=load_report_csv("impact_classification_model_comparison.csv")
    per_class=load_report_csv("impact_classification_per_class.csv")
    confusion=load_report_csv("impact_classification_confusion_matrices.csv")
    if comparison.empty or per_class.empty or confusion.empty:
        st.error("Paper evaluation outputs are missing. Run: python scripts/run_paper_experiments.py")
    else:
        comparison["algorithm"]=pd.Categorical(comparison["algorithm"],MODEL_ORDER,ordered=True)
        comparison=comparison.sort_values("algorithm")
        st.markdown("**Held-out model comparison**")
        shown=comparison.copy()
        for c in ["accuracy","macro_precision","macro_recall","macro_f1"]:
            if c in shown: shown[c]=(shown[c]*100).round(2)
        shown=shown.rename(columns={"algorithm":"Algorithm","accuracy":"Accuracy (%)","macro_precision":"Macro precision (%)","macro_recall":"Macro recall (%)","macro_f1":"Macro-F1 (%)","fit_predict_seconds":"Fit + predict (s)"})
        st.dataframe(shown,use_container_width=True,hide_index=True)
        best=comparison.loc[comparison["macro_f1"].idxmax()]
        st.success(f"Best overall model by macro-F1: {best['algorithm']} ({best['macro_f1']:.4f}). Decision Tree, Logistic Regression and Linear SVM are the three assessed classifiers.")
        selected=st.selectbox("Inspect confusion matrix",MODEL_ORDER,index=2)
        matrix_data=confusion[confusion.algorithm==selected]
        matrix=matrix_data.pivot(index="actual_class",columns="predicted_class",values="count").reindex(index=LABELS,columns=LABELS,fill_value=0)
        st.plotly_chart(px.imshow(matrix,text_auto=True,x=LABELS,y=LABELS,labels={"x":"Predicted class","y":"Actual class","color":"Records"},color_continuous_scale="Blues",aspect="auto",title=f"{selected}: confusion matrix"),use_container_width=True, key="plotly_chart_11")
        st.markdown(f"**Class-wise precision, recall and F1 — {selected}**")
        pc=per_class[per_class.algorithm==selected].copy()
        if not pc.empty:
            pc=pc[["class","precision","recall","f1-score","support"]].rename(columns={"class":"Class","precision":"Precision","recall":"Recall","f1-score":"F1-score","support":"Support"})
            for c in ["Precision","Recall","F1-score"]: pc[c]=(pc[c]*100).round(2)
            st.dataframe(pc,use_container_width=True,hide_index=True)
        st.info("Interpretation: the target is a researcher-defined proxy for recorded impact, not an official severity label. Results are retrospective classification, not pre-event forecasting or an emergency warning system. Macro-averaged metrics are important because the classes are imbalanced.")

with tab_clustering:
    st.subheader("Supplementary unsupervised clustering comparison")
    st.caption("K-Means, DBSCAN and Gaussian Mixture Model (GMM) are presented separately from the three supervised classifiers. These are exploratory impact-profile clusters, not official severity labels or classification predictions.")
    cluster_results = load_report_csv("three_clustering_algorithms_comparison.csv")
    if cluster_results.empty:
        st.error("Clustering results are missing. Check reports/three_clustering_algorithms_comparison.csv in the project package.")
    else:
        # Select the best tested configuration for each method by silhouette.
        best_rows = cluster_results.loc[cluster_results.groupby("algorithm")["silhouette"].idxmax()].copy()
        method_order = ["K-Means", "DBSCAN", "GMM"]
        best_rows["algorithm"] = pd.Categorical(best_rows["algorithm"], method_order, ordered=True)
        best_rows = best_rows.sort_values("algorithm")
        metric_cols = ["algorithm", "clusters_found", "silhouette", "davies_bouldin", "calinski_harabasz", "noise_fraction"]
        display = best_rows[metric_cols].copy()
        display["noise_fraction"] = (display["noise_fraction"] * 100).round(2)
        display = display.rename(columns={"algorithm":"Algorithm", "clusters_found":"Clusters found", "silhouette":"Silhouette (higher is better)", "davies_bouldin":"Davies-Bouldin (lower is better)", "calinski_harabasz":"Calinski-Harabasz (higher is better)", "noise_fraction":"Noise (%)"})
        st.markdown("**Best tested configuration for each clustering algorithm**")
        st.dataframe(display, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        for col, method in zip((c1, c2, c3), method_order):
            row = best_rows[best_rows.algorithm == method].iloc[0]
            if method == "K-Means":
                setting = f"k = {int(row['n_clusters'])}"
            elif method == "DBSCAN":
                setting = f"eps = {row['eps']}, min_samples = {int(row['min_samples'])}"
            else:
                setting = f"Components = {int(row['n_components'])}"
            col.metric(method, f"Silhouette {row['silhouette']:.3f}", setting)
        st.markdown("**Silhouette score across tested configurations**")
        plot_data = cluster_results.copy()
        plot_data["configuration"] = plot_data.apply(lambda r: (f"k={int(r['n_clusters'])}" if r['algorithm']=="K-Means" else (f"eps={r['eps']}" if r['algorithm']=="DBSCAN" else f"components={int(r['n_components'])}")), axis=1)
        st.plotly_chart(px.line(plot_data, x="configuration", y="silhouette", color="algorithm", markers=True, labels={"configuration":"Tested configuration", "silhouette":"Silhouette score", "algorithm":"Algorithm"}), use_container_width=True, key="plotly_chart_12")
        method_choice = st.selectbox("Inspect tested configurations", method_order, key="cluster_method_choice")
        detail = cluster_results[cluster_results.algorithm == method_choice].copy()
        detail_cols = [c for c in ["n_clusters", "eps", "min_samples", "n_components", "clusters_found", "silhouette", "davies_bouldin", "calinski_harabasz", "noise_fraction"] if c in detail.columns and detail[c].notna().any()]
        st.dataframe(detail[detail_cols].round(4), use_container_width=True, hide_index=True)
        st.info("Experimental setup: 5,000-record fitting sample and 2,500-record evaluation subset (random seed 42). Impact indicators were log1p-transformed, median-imputed and standardised. Negative values were treated as missing. The target class was not used to fit clusters. DBSCAN noise points are excluded from internal validity calculations and reported separately. Silhouette, Davies-Bouldin and Calinski-Harabasz are internal geometric metrics, not predictive accuracy.")
        st.caption("Interpretation: K-Means produced the strongest silhouette score in the tested configurations, but its two clusters must not be equated with Low and High impact categories. Cluster identifiers are arbitrary and do not rank severity.")

st.divider()
st.caption("Core supervised ML Lab: Decision Tree, Logistic Regression and Linear SVM. Supplementary unsupervised clustering is provided in its own Clustering Analysis tab. Contextual predictors are used in classification; target-defining impact fields are excluded from model features.")
