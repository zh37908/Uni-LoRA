#!/usr/bin/env python3
"""Reanalyse saved runs for revision-plan section 4. No model training or inference.
Run with the existing unilora_nlu Python environment. Output is self-contained
JSON/CSV plus generated LaTeX tables; figures consume these outputs, not LaTeX.
"""
from pathlib import Path
import csv
import itertools
import json
import sys
from collections import defaultdict
import numpy as np
from scipy.stats import t
import torch

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
NLU = REPO / 'NLU/peft/examples/sequence_classification'
OUT = PAPER / 'revision/existing_evidence'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(NLU))
import analyze_theory_e3 as e3
import analyze_theory_e4_e5 as e5
import summarize_theory_results as old_e1

torch.set_num_threads(2)

def save(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def csvout(name, rows):
    with (OUT / (name + '.csv')).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def rel(p):
    return str(Path(p).resolve().relative_to(REPO))

def fit(x, y, constrained=False):
    # Scaled slope c=a/n_min for stable fitting; y may be bootstrap x fractions.
    y = np.atleast_2d(y)
    X = np.column_stack([x, -np.ones_like(x)])
    coef = y @ np.linalg.pinv(X).T
    if constrained:
        candidates = np.stack([coef,
            np.column_stack([np.maximum(y @ x / (x @ x), 0), np.zeros(len(y))]),
            np.column_stack([np.zeros(len(y)), np.maximum(-y.mean(1), 0)]),
            np.zeros_like(coef)], axis=1)
        err = ((candidates @ X.T - y[:, None, :])**2).sum(2)
        err[(candidates < -1e-12).any(2)] = np.inf
        coef = candidates[np.arange(len(y)), err.argmin(1)]
        coef = np.maximum(coef, 0)
    return coef, coef @ X.T

def run_e1():
    rows = old_e1.collect(NLU / 'results_theory_e1_crossover')
    selected, source = {}, []
    for task in ['sst2', 'qnli', 'cola', 'mrpc']:
        rs = [r for r in rows if r['task'] == task]
        baseline = 'lora(lr=0.0004)' if task in ['sst2', 'cola'] else 'lora'
        rs = [r for r in rs if r['variant'] in [baseline, 'unilora']]
        groups = defaultdict(dict)
        for r in rs:
            key = (r['subset_ratio'], r['variant'])
            assert r['seed'] not in groups[key], ('duplicate seed', r)
            groups[key][r['seed']] = r
        points, paired = [], []
        for p in sorted({k[0] for k in groups}):
            a, b = groups[p, baseline], groups[p, 'unilora']
            assert set(a) == set(b) == set(range(5))
            diffs = []
            for s in range(5):
                assert a[s]['actual_train_size'] == b[s]['actual_train_size']
                assert a[s]['subset_seed'] == b[s]['subset_seed'] == 12345
                diff = a[s]['min_val_loss'] - b[s]['min_val_loss']
                diffs.append(diff)
                source.append(dict(task=task, fraction=p, seed=s,
                    n=a[s]['actual_train_size'], subset_seed=a[s]['subset_seed'],
                    lora_loss=a[s]['min_val_loss'], unilora_loss=b[s]['min_val_loss'], gap=diff,
                    lora_path=rel(a[s]['path']), unilora_path=rel(b[s]['path'])))
            half = float(t.ppf(.975, 4) * np.std(diffs, ddof=1) / np.sqrt(5))
            points.append(dict(fraction=p, n=a[0]['actual_train_size'], mean=float(np.mean(diffs)),
                               sd=float(np.std(diffs, ddof=1)), ci_low=float(np.mean(diffs)-half),
                               ci_high=float(np.mean(diffs)+half)))
            paired.append(diffs)
        n = np.array([r['n'] for r in points]); x = n.min()/n
        y = np.mean(paired, axis=1)
        # Enumerate all 5^5 ordered resamples of entire seed trajectories, retaining
        # pairing across methods AND dependence across sample-size fractions.
        ix = np.array(list(itertools.product(range(5), repeat=5)))
        boot_y = np.array(paired).T[ix].mean(1)
        fits = {}
        for mode in ['free', 'constrained']:
            coef, pred = fit(x, y, mode == 'constrained')
            bc, bp = fit(x, boot_y, mode == 'constrained')
            coef[:, 0] *= n.min(); bc[:, 0] *= n.min()
            residual = y - pred[0]
            boot_rmse = np.sqrt(np.mean((boot_y-bp)**2, axis=1))
            fits[mode] = dict(a=float(coef[0,0]), b=float(coef[0,1]),
                a_ci=np.quantile(bc[:,0],[.025,.975]).tolist(),
                b_ci=np.quantile(bc[:,1],[.025,.975]).tolist(),
                rmse=float(np.sqrt(np.mean(residual**2))), rmse_ci=np.quantile(boot_rmse,[.025,.975]).tolist(),
                r2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
                residuals=residual.tolist(), predictions=pred[0].tolist())
        selected[task] = dict(baseline=baseline, points=points, fits=fits)
        print('E1', task, fits, flush=True)
    save('e1', selected); csvout('e1_paired_runs', source)

def run_e3():
    items = e3.drop_legacy_lora(e3.scan(NLU/'results_theory_e3_bias_variance', 'roberta-large'))
    ref_items = [i for i in e3.drop_legacy_lora(e3.scan(NLU/'results_theory_e4_projection_sweep', 'roberta-large'))
                 if i['method']=='lora' and i['boot'] is None]
    result = {}
    for task in ['cola','mrpc']:
        its = [i for i in items if i['task']==task]
        r4 = [i for i in ref_items if i['task']==task]
        r64 = [i for i in its if i['method']=='lora_ref']
        best = max(i['score'] for i in r4+r64)
        kept64 = [i for i in r64 if i['score']>=.85*best]
        rp = {str(i['path']):e3.probs(e3.load_pt(i['path'])) for i in r4+kept64}
        refs = {'mixed':r4+kept64, 'rank4':r4, 'rank64':kept64}
        for j in range(len(refs['mixed'])):
            refs[f'leave_one_out_{j}'] = refs['mixed'][:j]+refs['mixed'][j+1:]
        means = {k:np.mean([rp[str(i['path'])] for i in v],axis=0) for k,v in refs.items()}
        desc = {k:[dict(path=rel(i['path']),score=i['score'],seed=i['seed']) for i in v] for k,v in refs.items()}
        methods = {}
        for method in ['lora','unilora','prolosa']:
            runs = [i for i in its if i['method']==method and i['boot'] is not None]
            grouped = defaultdict(dict)
            for i in runs:
                assert i['seed'] not in grouped[i['boot']]
                obj = e3.load_pt(i['path'])
                js = e3.sidecar(i['path'])
                assert js['args']['subset_seed']==i['boot']
                assert js['args']['seed']==i['seed']
                assert obj['eval_logits'].shape[-1]==2
                grouped[i['boot']][i['seed']] = e3.probs(obj)
            assert set(grouped)==set(range(101,106))
            assert all(set(v)=={0,1} for v in grouped.values())
            z = np.array([[grouped[b][s] for s in [0,1]] for b in sorted(grouped)])
            nex = z.shape[-1]//2
            bm=z.mean(1); gm=bm.mean(0)
            vo=float(np.sum((z-bm[:,None])**2)/(5*(2-1)*nex))
            vb=float(np.sum((bm-gm)**2)/((5-1)*nex))
            vs=vb-vo/2
            out, _ = e3.decompose({b:list(v.values()) for b,v in grouped.items()},means['mixed'],None,nex)
            assert np.allclose([out['V_opt'],out['V_stat']],[vo,max(vs,0)])
            # Crossed random-effects sensitivity: same seed labels recur over b.
            interaction = z-bm[:,None]-z.mean(0)[None,:]+gm
            vi=float(np.sum(interaction**2)/((5-1)*(2-1)*nex))
            ms_seed=float(5*np.sum((z.mean(0)-gm)**2)/((2-1)*nex))
            out.update(V_stat_signed=vs,
                crossed_V_stat_signed=vb-vi/2,
                crossed_V_seed_signed=(ms_seed-vi)/5,
                crossed_V_interaction=vi,
                bias_by_reference={k:float(np.sum((gm-v)**2)/nex) for k,v in means.items()},
                paths=[rel(i['path']) for i in runs])
            methods[method]=out
        result[task]=dict(methods=methods,references=desc,
            excluded_references=[dict(path=rel(i['path']),score=i['score']) for i in r64 if i not in kept64])
        print('E3',task,{m:{k:v for k,v in o.items() if k not in ['paths','bias_by_reference']} for m,o in methods.items()},flush=True)
    save('e3', result)

def run_e5():
    roots = ['results_theory_e1_crossover','results_theory_e3_bias_variance',
             'results_theory_e4_remaining_glue','results_theory_e4_projection_sweep']
    paths=[]
    for root in roots:
        paths.extend(sorted((NLU/root).rglob('*_theory.pt')))
    meta=[dict(path=p,method=e5.classify_method(p)[0],task=p.parts[p.parts.index('roberta-large')+1],
               dedicated=bool(e3.re.search(r'/lora(?:_ref)?_r\d+_lr',str(p))))
          for p in paths if 'roberta-large' in p.parts]
    dedicated={r['task'] for r in meta if r['method']=='lora' and r['dedicated']}
    meta=[r for r in meta if not(r['method']=='lora' and r['task'] in dedicated and not r['dedicated'])]
    result={}
    for task in ['cola','mrpc','qnli','rte','stsb']:
        loras=[r for r in meta if r['task']==task and r['method']=='lora']
        pro=next(r for r in meta if r['task']==task and r['method']=='prolosa'
                 and '/boot_' not in str(r['path'])
                 and ('/p1/' in str(r['path']) or 'results_theory_e4_remaining_glue' in str(r['path'])))
        obj=e3.load_pt(pro['path']); names=obj['module_names']
        D=obj['delta_theta'].numel()
        full_sum=np.zeros(D); old_sum=np.zeros(D); full_paths=[]
        for r in loras:
            o=e3.load_pt(r['path']); assert o['module_names']==names and o['rank']==4
            d=o['delta_theta'].double().numpy(); old_sum+=d
            if o['train_subset_ratio']==1 and '/boot_' not in str(r['path']):
                full_sum+=d; full_paths.append(rel(r['path']))
        target=full_sum/len(full_paths)
        legacy_target=old_sum/len(loras)
        hobj=e3.load_pt(NLU/f'fisher_diag/{task}_fisher.pt')
        h=hobj['fisher_diag'].double().numpy()
        assert hobj['module_names']==names and h.shape==target.shape and np.all(h>=0)
        assignment=e5.reconstruct_assignment(obj['index_meta'],names,D)
        L=int(obj['theta_d_length']); assert np.all((assignment>=0)&(assignment<L))
        support=e5.snip_support_in_delta_order(obj)
        assert support is not None and support.shape==(D,)
        K=int(support.sum()); assert K>0
        # Check sparse offsets form a bijection onto the original-coordinate candidates.
        offsets=np.concatenate([obj['index_meta'][n][k].reshape(-1).numpy() for n in names for k in ['offsets_A','offsets_B']])
        assert np.array_equal(np.sort(offsets),np.arange(D))
        qw=np.bincount(assignment,weights=h*target,minlength=L)
        hw=np.bincount(assignment,weights=h,minlength=L)
        weighted_mean=np.divide(qw,hw,out=np.zeros_like(qw),where=hw>0)
        qh=target-weighted_mean[assignment]
        qe=e5.off_subspace_residual(target,assignment,L)
        le=e5.off_subspace_residual(legacy_target,assignment,L)
        def stats(q):
            en=h*q*q; tot=en.sum()
            return dict(energy=float(tot),oracle=e5.topk_energy_ratio(en,K),snip=float(en[support].sum()/tot))
        es,hs,ls=stats(qe),stats(qh),stats(le)
        # Exact H-weighted approximation-bias recovery after jointly refitting
        # bucket coefficients and selected coordinates (still a target/Fisher proxy).
        keep=~support
        sw=np.bincount(assignment[keep],weights=h[keep],minlength=L)
        sy=np.bincount(assignment[keep],weights=(h*target)[keep],minlength=L)
        mu=np.divide(sy,sw,out=np.zeros_like(sy),where=sw>0)
        hyb_energy=float(np.sum(h[keep]*(target[keep]-mu[assignment[keep]])**2))
        gain=1-hyb_energy/hs['energy']
        assert hs['energy']<=es['energy']*(1+1e-10) and gain>=hs['snip']-1e-10
        # 10,000 uniform size-K supports, without replacement, from all D candidates.
        # Exact finite-population mean and SD accompany the Monte Carlo check.
        rng=np.random.default_rng(20260913+['cola','mrpc','qnli','rte','stsb'].index(task))
        en=h*qe*qe; en/=en.sum()
        draws=np.array([en[rng.choice(D,K,replace=False)].sum() for _ in range(10000)])
        exact_sd=float(np.sqrt(K*(D-K)/(D-1)*np.var(en)))
        out=dict(D=D,d=L,K=K,support_path=rel(pro['path']),support_seed=int(obj['seed']),
            support_train_fraction=obj['train_subset_ratio'],full_target_paths=full_paths,
            pooled_target_sensitivity_paths=[rel(r['path']) for r in loras],fisher_path=rel(NLU/f'fisher_diag/{task}_fisher.pt'),
            fisher_metadata={k:hobj[k] for k in ['num_batches','batch_size','split','rank','artifact']},
            euclidean_then_fisher=es,h_projection=hs,h_joint_recovery=gain,
            pooled_target_sensitivity=ls,random_expected=K/D,enrichment=es['snip']/(K/D),
            bare_enrichment=float((qe[support]**2).sum()/(qe**2).sum()/(K/D)),
            random_mean=float(draws.mean()),random_sd=float(draws.std(ddof=1)),random_exact_sd=exact_sd,
            random_q025=float(np.quantile(draws,.025)),random_q975=float(np.quantile(draws,.975)),
            random_mc_se=exact_sd/100,random_exceedances=int(np.sum(draws>=es['snip'])),n_draws=len(draws))
        result[task]=out
        print('E5',task,{k:v for k,v in out.items() if not k.endswith('paths')},flush=True)
    save('e5',result)

if __name__=='__main__':
    for exp in sys.argv[1:] or ['e1','e3','e5']:
        globals()['run_'+exp]()
