"""Audit existing synthetic aggregates and enforce fixed-target decomposition.
No training. Run with /tmp/iclr-layout-env/bin/python (numpy only required).
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[2]; S=R/'synthetic';O=R/'ICLR_2027/revision/existing_evidence'
sys.path.insert(0,str(S))
from fixed_target import fixed_target_decomposition,target_id
from synthetic_powerlaw_noise_biasvar import stable_seed,orthonormal_projection,summarize_estimates
from synthetic_contaminated_gaussian_noise_biasvar import make_contaminated_gaussian_theta
from synthetic_gaussian_noise_biasvar import make_gaussian_theta
from synthetic_validate_theory import make_problem

# A perfect estimator of each of two DIFFERENT targets has zero risk.
# Pooling those estimates against their average invents positive variance.
# The interface must reject stacked targets and inconsistent per-trial risks.
for target,risks in [(np.array([[1.,0.],[-1.,0.]]),None),(np.zeros(2),[0.,0.])]:
    try: fixed_target_decomposition([[1.,0.],[-1.,0.]],target,risks)
    except ValueError: pass
    else: raise AssertionError('Mixed-target decomposition was accepted')
a=np.array([[2.,1.],[0.,1.]])
assert np.allclose(fixed_target_decomposition(a,np.array([1.,0.])),[1.,.5,.5])
assert np.allclose(summarize_estimates(list(a),[1.,1.],np.array([1.,0.])),[1.,0.,.5,.5])
# A learner-projection sweep must keep teacher target and teacher projection fixed.
problems=[make_problem(64,4,.8,5,1.,123,hidden_p=True,pm_mode=m,pm_angle_deg=ang)
          for m,ang in [('rotated',0),('rotated',15),('rotated',30),('independent',0)]]
assert all(np.array_equal(p.theta_star,problems[0].theta_star) and
           np.array_equal(p.P_teacher,problems[0].P_teacher) for p in problems)

checks={}
files=[('results_matched_budget_baselines/matched_budget_results.csv','bias','variance'),
       ('results_spike_slab_budget_matched_theory/spike_slab_results.csv','bias','variance'),
       ('results_gaussian_synthetic_theory_std_0.125/gaussian_results.csv','bias','variance'),
       ('results_transformer_lab_level2_v2_full/summary.csv','bias_fn','variance_fn')]
for name,b,v in files:
    rows=list(csv.DictReader((S/name).open()))
    errors=[abs(float(r['risk_mean'])-float(r[b])-float(r[v])) for r in rows]
    assert max(errors)<1e-8,(name,max(errors))
    checks[name]=dict(rows=len(rows),max_identity_error=max(errors),sha256=hashlib.sha256((S/name).read_bytes()).hexdigest())
# Recover the fixed sequence targets from the original generation rules/configs.
c=json.loads((S/'results_matched_budget_baselines/config.json').read_text())
P=orthonormal_projection(c['D'],c['d'],np.random.default_rng(stable_seed(c['seed'],'contaminated_projection')))
theta,_=make_contaminated_gaussian_theta(c['D'],c['spike_std'],c['slab_probability'],c['slab_std']/c['spike_std'],c['slab_df'],np.random.default_rng(stable_seed(c['seed'],'contaminated_problem',c['slab_probability'])))
assert np.isclose(np.linalg.norm(theta),c['theta_norm'])
q=theta-P@(P.T@theta)
checks['concentrated_target']=dict(target_id=target_id(theta),theta_norm=float(np.linalg.norm(theta)),off_energy=float(q@q),noise_threshold=float(np.sqrt(c['n_eff']*(q@q)/(c['D']-c['d']))))
c=json.loads((S/'results_gaussian_synthetic_theory_std_0.125/config.json').read_text())
rng=np.random.default_rng(stable_seed(c['seed'],'gaussian_problem'));P=orthonormal_projection(c['D'],c['d'],rng)
theta,_=make_gaussian_theta(c['D'],c['source_std'],rng);q=theta-P@(P.T@theta)
assert np.isclose(np.linalg.norm(theta),c['realized_theta_norm'])
checks['diffuse_target']=dict(target_id=target_id(theta),theta_norm=float(np.linalg.norm(theta)),off_energy=float(q@q),noise_threshold=float(np.sqrt(c['n_eff']*(q@q)/(c['D']-c['d']))))
# Validate Transformer merges by full cell key, never pooling profiles or n/noise.
p=S/'results_transformer_lab_level2_v2_full'
trials=list(csv.DictReader((p/'trials.csv').open()));summary=list(csv.DictReader((p/'summary.csv').open()))
key=lambda r:(r['profile'],int(r['n']),float(r['noise_std']),r['method'])
groups=defaultdict(list)
for r in trials:groups[key(r)].append(r)
assert len({key(r) for r in summary})==len(summary)==len(groups)
for r in summary:
    g=groups[key(r)];seeds=[int(x['seed']) for x in g]
    assert len(set(seeds))==len(g)==int(r['n_seeds'])
    assert len(g)==(2 if float(r['noise_std'])==0 else 20)
    vals=np.array([float(x['excess_risk']) for x in g])
    assert np.allclose([vals.mean(),vals.std(ddof=1)],[float(r['risk_mean']),float(r['risk_std'])])
meta=list(csv.DictReader((p/'metadata.csv').open()))
for profile in {m['profile'] for m in meta}:
    ms=[m for m in meta if m['profile']==profile]
    for field in ['theta_norm','plant_scale','weight_delta_norm','energy']:
        assert len({m[field] for m in ms})==1
checks['transformer_merge']=dict(trials=len(trials),cells=len(summary),
    main_cells=sum(float(r['noise_std'])>0 for r in summary),
    calibration_cells=sum(float(r['noise_std'])==0 for r in summary),
    note='Main and calibration cells are disjoint. Source .py and raw predictions are absent; retained runner bytecode and logs show planting outside trial loops. Aggregate identities do not independently prove target equality.')
# Verify paper sequence CSV copies are byte-identical to their synthetic sources.
for source,dest in [('results_matched_budget_baselines/matched_budget_results.csv','matched_budget_results.csv'),('results_gaussian_synthetic_theory_std_0.125/gaussian_results.csv','gaussian_std0125_results.csv')]:
    assert (S/source).read_bytes()==(R/'ICLR_2027/figures/data'/dest).read_bytes()
checks['guards']='fixed target identity, mixed-target rejection, unchanged teacher across projection modes, raw-run merge keys: passed'
(O/'fixed_target_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
