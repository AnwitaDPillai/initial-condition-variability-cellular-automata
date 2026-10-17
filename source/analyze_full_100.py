"""Preregistered grouped-CV analysis; run only after full data are frozen."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
ROOT=Path(__file__).resolve().parent
runs=pd.read_csv(ROOT/'data'/'full_runs_100.csv')
summary=runs.groupby(['rule','equivalence_class']).agg(
    lz_mean=('lz_mean_rows','mean'), lz_sd=('lz_mean_rows','std'),
    density_temporal_sd_mean=('density_temporal_sd','mean')).reset_index()
feat=runs.drop_duplicates('rule')[['rule','equivalence_class','lambda','sensitivity','influence_left','influence_center','influence_right']].copy()
feat['lambda_sym'] = np.minimum(feat['lambda'], 1.0-feat['lambda'])
feat['influence_lr_mean'] = (feat['influence_left'] + feat['influence_right']) / 2.0
feat['influence_lr_absdiff'] = np.abs(feat['influence_left'] - feat['influence_right'])
d=summary.merge(feat,on=['rule','equivalence_class'])
y=np.log(d['lz_sd'].fillna(0).to_numpy()+0.001)
groups=d['equivalence_class'].to_numpy()
cv=GroupKFold(n_splits=5)
models={
    'null_training_mean': None,
    'baseline_lambda_sensitivity': ['lambda_sym','sensitivity'],
    'enriched_symmetry_invariant_influences': ['lambda_sym','sensitivity','influence_center','influence_lr_mean','influence_lr_absdiff'],
}
results=[]
preds={}
for name,cols in models.items():
    if cols is None:
        pred=np.empty_like(y)
        for train,test in cv.split(d,y,groups): pred[test]=y[train].mean()
    else:
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        pred=cross_val_predict(model,d[cols],y,groups=groups,cv=cv)
    preds[name]=pred
    results.append({'model':name,'rmse_log_sd':mean_squared_error(y,pred)**0.5,
                    'mae_log_sd':mean_absolute_error(y,pred),'r2_log_sd':r2_score(y,pred)})
res=pd.DataFrame(results)
res.to_csv(ROOT/'data'/'grouped_cv_results_100.csv',index=False)
out=d[['rule','equivalence_class','lz_mean','lz_sd','density_temporal_sd_mean']].copy()
for name,pred in preds.items(): out['pred_'+name]=pred
out.to_csv(ROOT/'data'/'grouped_cv_predictions_100.csv',index=False)
print('5-fold GroupKFold; groups are 88 exact symmetry classes; predictors are symmetry-invariant; target log(sample SD + 0.001)')
print(res.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
