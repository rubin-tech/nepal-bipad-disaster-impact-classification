import pandas as pd, numpy as np, time, json
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
ROOT=Path(__file__).resolve().parents[1]
p=ROOT
df=pd.read_csv(p/'data/raw/Nepal_Disaster_Data_2020_2026.csv', low_memory=False)
cols=['peopleDeathCount','peopleMissingCount','peopleInjuredCount','peopleAffectedCount','familyAffectedCount','infrastructureDestroyedCount','infrastructureAffectedHouseCount','infrastructureDestroyedRoadCount','infrastructureDestroyedBridgeCount','infrastructureEconomicLoss','agricultureEconomicLoss','estimatedLoss']
X=df[[c for c in cols if c in df]].apply(pd.to_numeric,errors='coerce')
loss=X['estimatedLoss'].notna().astype(float) if 'estimatedLoss' in X else pd.Series(0.,index=X.index)
for c in X:
 X[c]=X[c].where(X[c]>=0)
 X[c]=np.log1p(X[c])
# no missingness indicator
X=X.dropna(axis=1,how='all'); X=X.loc[:,X.nunique(dropna=True)>1]
X=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))
rng=np.random.default_rng(42); fit_idx=np.sort(rng.choice(len(df),size=min(5000,len(df)),replace=False)); eval_local=np.sort(rng.choice(len(fit_idx),size=min(2500,len(fit_idx)),replace=False)); xf=X[fit_idx]; xe=xf[eval_local]
rows=[]
def add(name,params,labels,elapsed):
 labels=np.asarray(labels); u=np.unique(labels); noise=labels==-1 if name=='DBSCAN' else np.zeros(len(labels),bool); keep=~noise; labs=labels[keep]; xx=xe[keep]; uu=np.unique(labs)
 sil=db=ch=np.nan
 if len(uu)>1 and len(uu)<len(labs):
  sil=float(silhouette_score(xx,labs)); db=float(davies_bouldin_score(xx,labs)); ch=float(calinski_harabasz_score(xx,labs))
 rows.append({'algorithm':name,**params,'silhouette':sil,'davies_bouldin':db,'calinski_harabasz':ch,'clusters_found':len(uu),'noise_fraction':float(noise.mean()),'fit_seconds':round(elapsed,3),'fit_sample_n':len(xf),'evaluation_sample_n':len(xe),'evaluation_non_noise_n':len(labs)})
# KMeans search
for k in [2,3,4,5,6]:
 m=KMeans(n_clusters=k,random_state=42,n_init=10); t=time.perf_counter(); m.fit(xf); dt=time.perf_counter()-t; add('K-Means',{'n_clusters':k},m.predict(xe),dt)
# DBSCAN configs
for eps in [0.8,1.0,1.2,1.5,2.0]:
 m=DBSCAN(eps=eps,min_samples=10,n_jobs=-1); t=time.perf_counter(); labs=m.fit_predict(xf); dt=time.perf_counter()-t; add('DBSCAN',{'eps':eps,'min_samples':10},labs[eval_local],dt)
# GMM search
for k in [2,3,4,5,6]:
 m=GaussianMixture(n_components=k,covariance_type='diag',random_state=42,n_init=2,reg_covar=1e-5,max_iter=200); t=time.perf_counter(); m.fit(xf); dt=time.perf_counter()-t; add('GMM',{'n_components':k},m.predict(xe),dt)
out=pd.DataFrame(rows); out.to_csv(p/'reports/three_clustering_algorithms_comparison.csv',index=False)
print(out.to_string(index=False))
print('ROWS',len(df),'FEATURES',list(X.shape))
