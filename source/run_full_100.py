"""Run the exploratory all-rule simulation protocol."""
from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.automaton import evolve, random_initial
from src.features import rule_features
from src.metrics import trajectory_metrics
from src.symmetry import all_equivalence_classes

MASTER_SEED=20261017
WIDTH=200
STEPS=200
N_SEEDS=100
classes=all_equivalence_classes()
assert len(set(classes.values())) == 88, f"Expected 88 symmetry classes, got {len(set(classes.values()))}"
outfile=ROOT/'data'/'full_runs_100.csv'
rows=[]
t0=time.time()
checkpoint=ROOT/'data'/'full_runs_100_checkpoint.csv'
start_rule=0
if checkpoint.exists():
    prior=pd.read_csv(checkpoint)
    rows=prior.to_dict('records')
    start_rule=int(prior['rule'].max())+1 if len(prior) else 0
    print(f'resuming at rule {start_rule}',flush=True)
for rule in range(start_rule,256):
    feat=rule_features(rule)
    for seed_idx in range(N_SEEDS):
        seed=int(np.random.SeedSequence([MASTER_SEED,rule,seed_idx]).generate_state(1)[0])
        initial=random_initial(WIDTH,seed)
        st=evolve(rule,initial,STEPS)
        m=trajectory_metrics(st)
        rows.append({**feat,'equivalence_class':classes[rule],'seed_index':seed_idx,'rng_seed':seed,
                     'width':WIDTH,'steps':STEPS,**m})
    if (rule+1) % 8 == 0:
        pd.DataFrame(rows).to_csv(checkpoint,index=False)
        print(f'completed {rule+1}/256 rules in {time.time()-t0:.1f}s', flush=True)
df=pd.DataFrame(rows)
df.to_csv(outfile,index=False)
if checkpoint.exists(): checkpoint.unlink()
summary=df.groupby(['rule','equivalence_class']).agg(
    n=('seed_index','count'), lz_mean=('lz_mean_rows','mean'),
    lz_sd_across_seeds=('lz_mean_rows','std'), lz_median=('lz_mean_rows','median'),
    lz_q10=('lz_mean_rows',lambda x:x.quantile(.1)), lz_q90=('lz_mean_rows',lambda x:x.quantile(.9)),
    density_mean=('density_mean','mean'), density_temporal_sd_mean=('density_temporal_sd','mean'),
    density_temporal_sd_across_seeds=('density_temporal_sd','std')).reset_index()
summary.to_csv(ROOT/'data'/'full_summary_100.csv',index=False)
config={'master_seed':MASTER_SEED,'width':WIDTH,'steps':STEPS,'seeds_per_rule':N_SEEDS,
        'rules':list(range(256)),'boundary':'periodic','initial_state':'iid Bernoulli(0.5)',
        'primary_outcome':'per-run mean normalized LZ78 phrase count C(n)*log2(n)/n averaged over evolved rows t=1..T',
        'variability_target':'sample standard deviation (ddof=1) across seeds of per-run primary outcome',
        'equivalence_classes':88,'elapsed_seconds':round(time.time()-t0,2)}
(ROOT/'data'/'full_config_100.json').write_text(json.dumps(config,indent=2))
print('Saved',outfile,'rows=',len(df),'elapsed=',round(time.time()-t0,1),'s')
