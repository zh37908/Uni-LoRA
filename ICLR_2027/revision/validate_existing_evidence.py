"""Independent numerical checks on paired fits, projections, and rendered values."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

P=Path(__file__).resolve().parent/'existing_evidence'
e1,e3,e5=[json.loads((P/f'{x}.json').read_text()) for x in ['e1','e3','e5']]
pairs=list(csv.DictReader((P/'e1_paired_runs.csv').open()))
for task,o in e1.items():
    y=np.array([q['mean'] for q in o['points']]);n=np.array([q['n'] for q in o['points']])
    X=np.column_stack([n.min()/n,-np.ones_like(n)])
    c,residual=nnls(X,y)
    assert np.allclose([c[0]*n.min(),c[1]],[o['fits']['constrained']['a'],o['fits']['constrained']['b']])
    assert np.isclose(residual/np.sqrt(len(y)),o['fits']['constrained']['rmse'])
    for q in o['points']:
        rs=[r for r in pairs if r['task']==task and float(r['fraction'])==q['fraction']]
        assert len(rs)==5 and len({r['seed'] for r in rs})==5
        assert np.isclose(np.mean([float(r['gap']) for r in rs]),q['mean'])
        assert q['ci_low']<=q['mean']<=q['ci_high']
for task,o in e3.items():
    expected=['lora','unilora','prolosa'] if task=='cola' else ['lora','prolosa','unilora']
    for ref in o['references']:
        assert sorted(o['methods'],key=lambda m:o['methods'][m]['bias_by_reference'][ref])==expected
    for m,v in o['methods'].items():
        assert np.isclose(v['V_total'],v['V_opt']+v['V_stat'])
        assert np.isclose(v['V_stat'],max(v['V_stat_signed'],0))
for task,o in e5.items():
    assert o['support_train_fraction']==1
    assert np.isclose(o['random_expected'],o['K']/o['D'])
    assert abs(o['random_mean']-o['random_expected'])<2*o['random_mc_se']
    assert o['h_projection']['energy']<=o['euclidean_then_fisher']['energy']
    assert o['h_joint_recovery']>=o['h_projection']['snip']
# Independent dense weighted least-squares check of the bucket formulas on a
# small anisotropic problem with an empty-weight bucket and coupled support.
rng=np.random.default_rng(93);D,d=24,4
assignment=np.arange(D)%d;h=rng.uniform(.1,3,D);h[assignment==3]=0
z=rng.normal(size=D);support=np.array([0,1,4,9,14])
Pmat=np.eye(d)[assignment];Emat=np.eye(D)[:,support]
def dense_error(A):
    coef=np.linalg.lstsq(np.sqrt(h)[:,None]*A,np.sqrt(h)*z,rcond=None)[0]
    return np.sum(h*(z-A@coef)**2)
def bucket_error(keep):
    sums=np.bincount(assignment[keep],weights=h[keep]*z[keep],minlength=d)
    weights=np.bincount(assignment[keep],weights=h[keep],minlength=d)
    means=np.divide(sums,weights,out=np.zeros(d),where=weights>0)
    return np.sum(h[keep]*(z[keep]-means[assignment[keep]])**2)
keep=np.ones(D,dtype=bool);assert np.isclose(dense_error(Pmat),bucket_error(keep))
keep[support]=False;assert np.isclose(dense_error(np.column_stack([Pmat,Emat])),bucket_error(keep))
main=(P.parents[1]/'sections/main.tex').read_text()
assert (P/'e5_main_rows.tex').read_text().strip() in main
print('PASS: paired means; NNLS constrained fits; reference ordering; variance clipping; random null; dense H-projection/joint recovery; manuscript table values')
