from pathlib import Path
import sys,time
import numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
ROOT=Path('/mnt/data/eca_project');sys.path.insert(0,str(ROOT))
from src.automaton import evolve,random_initial
from src.metrics import normalized_lz78,damage_curve
RULES=[12,196,204,221,207,220,100,68,69,206];WIDTHS=[200,400];T=200;N_TRAIN=100;N_TEST=200;MASTER=20261021
PRIMARY='density_mean';SECONDARY=['density_temporal_sd','damage_auc_50']
def seed_for(w,r,split,j):return int(np.random.SeedSequence([MASTER,w,r,split,j]).generate_state(1)[0])
def traj(r,w,seed):
 x=random_initial(w,seed);grid=evolve(r,x,T);z=grid[1:];d=z.mean(axis=1);damage=damage_curve(r,x,50)[1:]
 return {'rule':r,'width':w,'seed':seed,'lz_mean_rows':float(np.mean([normalized_lz78(row) for row in z])),'density_mean':float(d.mean()),'density_temporal_sd':float(d.std()),'damage_auc_50':float(damage.mean())}
def smd(a,b):
 den=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2);return float((np.mean(a)-np.mean(b))/den) if den>1e-12 else np.nan
start=time.time();train=[];test=[]
for w in WIDTHS:
 for r in RULES:
  for j in range(N_TRAIN):train.append(traj(r,w,seed_for(w,r,0,j)))
  for j in range(N_TEST):test.append(traj(r,w,seed_for(w,r,1,j)))
  print('completed',r,w,flush=True)
tr=pd.DataFrame(train);te=pd.DataFrame(test);tr.to_csv(ROOT/'data/width_specific_component_training.csv',index=False)
mods=[];labs=[]
for (r,w),gt in tr.groupby(['rule','width']):
 gm=GaussianMixture(2,random_state=731,n_init=20,reg_covar=1e-5).fit(gt.lz_mean_rows.to_numpy().reshape(-1,1));order=np.argsort(gm.means_.ravel());g=te[(te.rule==r)&(te.width==w)].copy();post=gm.predict_proba(g.lz_mean_rows.to_numpy().reshape(-1,1))[:,order[1]];g['high_lz_posterior']=post;g['component']=(post>=.5).astype(int);g['confidence']=np.maximum(post,1-post);g['confident']=g.confidence>=.8;labs.append(g)
 mods.append({'rule':r,'width':w,'train_n':len(gt),'gmm_low_mean':gm.means_.ravel()[order[0]],'gmm_high_mean':gm.means_.ravel()[order[1]],'component_gap':gm.means_.ravel()[order[1]]-gm.means_.ravel()[order[0]],'low_weight':gm.weights_[order[0]],'high_weight':gm.weights_[order[1]],'converged':gm.converged_,'test_n':len(g),'test_low_n':int((g.component==0).sum()),'test_high_n':int((g.component==1).sum()),'confident_fraction':float(g.confident.mean())})
lab=pd.concat(labs,ignore_index=True);lab.to_csv(ROOT/'data/width_specific_component_heldout.csv',index=False);pd.DataFrame(mods).to_csv(ROOT/'data/width_specific_component_models.csv',index=False)
out=[]
for (r,w),g in lab.groupby(['rule','width']):
 for metric in [PRIMARY]+SECONDARY:
  for subset,gg in [('all_assigned',g),('confident_only',g[g.confident])]:
   lo=gg.loc[gg.component==0,metric].to_numpy();hi=gg.loc[gg.component==1,metric].to_numpy()
   if len(lo)>=10 and len(hi)>=10:
    out.append({'rule':r,'width':w,'metric':metric,'subset':subset,'n_low':len(lo),'n_high':len(hi),'mean_low':lo.mean(),'mean_high':hi.mean(),'difference_high_minus_low':hi.mean()-lo.mean(),'smd':smd(hi,lo),'p_raw':float(mannwhitneyu(hi,lo,alternative='two-sided').pvalue)})
res=pd.DataFrame(out);res['p_bh']=np.nan
for _,ix in res.groupby(['width','metric','subset']).groups.items():res.loc[ix,'p_bh']=multipletests(res.loc[ix,'p_raw'],method='fdr_bh')[1]
res.to_csv(ROOT/'data/width_specific_component_heldout_tests.csv',index=False)
summary=[]
for (w,m,s),g in res.groupby(['width','metric','subset']):summary.append({'width':w,'metric':m,'subset':s,'rules_tested':len(g),'rules_BH_p_lt_0.05':int((g.p_bh<.05).sum()),'median_abs_SMD':g.smd.abs().median(),'median_abs_difference':g.difference_high_minus_low.abs().median()})
pd.DataFrame(summary).to_csv(ROOT/'data/width_specific_component_summary.csv',index=False)
rob=[]
for m in [PRIMARY]+SECONDARY:
 for s in ['all_assigned','confident_only']:
  g=res[(res.metric==m)&(res.subset==s)].pivot(index='rule',columns='width',values='difference_high_minus_low')
  if 200 in g.columns and 400 in g.columns:
   v=g[[200,400]].dropna();rob.append({'metric':m,'subset':s,'n_rules_paired':len(v),'same_direction_fraction':float((np.sign(v[200])==np.sign(v[400])).mean()) if len(v) else np.nan,'spearman_width_effects':float(v[200].corr(v[400],method='spearman')) if len(v)>2 else np.nan})
pd.DataFrame(rob).to_csv(ROOT/'data/width_specific_component_width_robustness.csv',index=False)
report=f'''# Width-specific held-out replication of LZ-component meaning

## Frozen protocol
For each of {len(RULES)} previously selected rules ({RULES}) and widths {WIDTHS}, independently generated {N_TRAIN} training trajectories and {N_TEST} held-out trajectories per rule-width (master seed {MASTER}). Initial states are Bernoulli(0.5), boundaries periodic, T={T}. A two-component Gaussian mixture is fit separately for every rule and width on training LZ scores only; held-out trajectories are assigned by posterior to the lower-/higher-LZ-mean component. No held-out independent outcomes influence fitting.

**Primary outcome:** mean density across evolved rows. **Secondary outcomes:** temporal SD of row density and single-cell damage AUC over updates 1–50. Within-rule/width comparisons use two-sided Mann–Whitney tests; Benjamini–Hochberg correction is across 10 rules within each width × metric × subset. Report both all hard assignments and a sensitivity analysis restricted to posterior confidence >=0.8.

## Results
See `data/width_specific_component_summary.csv`, `data/width_specific_component_heldout_tests.csv`, model parameters, and width robustness file. A component association does not establish distinct attractors: a mixture can partition a continuous or skewed distribution. Stronger evidence requires meaningful effects, multiplicity-adjusted support on the primary and at least one secondary outcome, adequate assignment confidence/balance, and cross-width directional consistency.

## Limitations
This is a post-hoc replication protocol, not externally preregistered. Held-out trajectories are independent of mixture fitting, but rules were selected in earlier work. Failure to reject is not evidence of equivalence. Damage AUC is specific to a single-cell perturbation and 50-update horizon.
'''
(ROOT/'FOLLOWUP_WIDTH_SPECIFIC_REPORT.md').write_text(report)
print('\nSUMMARY\n',pd.DataFrame(summary).to_string(index=False));print('\nMODELS\n',pd.DataFrame(mods)[['rule','width','component_gap','low_weight','high_weight','test_low_n','test_high_n','confident_fraction']].to_string(index=False));print('\nROBUSTNESS\n',pd.DataFrame(rob).to_string(index=False));print('Elapsed',round(time.time()-start,1),'seconds')
