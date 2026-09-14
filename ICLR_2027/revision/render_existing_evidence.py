"""Generate manuscript tables from audited JSON; no fitting or rounding inputs."""
import json
from pathlib import Path
P = Path(__file__).resolve().parent/'existing_evidence'
E1,E3,E5=[json.loads((P/f'{e}.json').read_text()) for e in ['e1','e3','e5']]
NAMES={'sst2':'SST-2','qnli':'QNLI','cola':'CoLA','mrpc':'MRPC','rte':'RTE','stsb':'STS-B'}

def table(name, caption, cols, header, rows, size='small'):
    s='\n'.join([r'\begin{table*}[t]',r'\centering'+chr(92)+size,
        r'\caption{'+caption+'}',r'\label{tab:'+name+'}',
        r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule',
        *[r+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])
    (P/(name+'.tex')).write_text(s)
rows=[]
for task,o in E1.items():
    for q in o['points']:
        rows.append(f"{NAMES[task]} & {q['fraction']*100:g}\\% & {q['n']} & {q['mean']:.4f} & [{q['ci_low']:.4f}, {q['ci_high']:.4f}]")
table('e1_paired_ci','Paired E1 mean loss gaps and pointwise 95\\% Student-$t$ intervals (five training seeds, four degrees of freedom). The training subset is fixed at each fraction; these intervals do not measure uncertainty over new training sets or new dev sets.','lrrrr','Task & Fraction & $n$ & Mean gap & 95\\% interval',rows)
rows=[]
for task,o in E1.items():
    for mode,f in o['fits'].items():
        a,b=f['a_ci'],f['b_ci']
        rows.append(f"{NAMES[task]} & {'Free' if mode=='free' else 'Constrained'} & {f['a']:.2f} [{a[0]:.2f}, {a[1]:.2f}] & {f['b']:.4f} [{b[0]:.4f}, {b[1]:.4f}] & {f['rmse']:.4f} & {f['r2']:.3f}")
table('e1_paired_fit','Unweighted fits of paired mean gaps to $a/n-b$. Brackets are descriptive 95\\% percentile intervals from all $5^5=3125$ resamples of complete seed trajectories. Constrained fits impose $a,b\\geq0$; boundary intervals such as $[0,0]$ are a consequence of the constraint and the five observed seeds.','llrrrr','Task & Fit & $a$ [interval] & $b$ [interval] & RMSE & $R^2$',rows,'scriptsize')
rows=[]
for task,o in E3.items():
    for m,label in [('lora','LoRA'),('unilora','Uni-LoRA'),('prolosa','ProLoSA')]:
        b=o['methods'][m]['bias_by_reference'];loo=[v for k,v in b.items() if k.startswith('leave')]
        rows.append(f"{NAMES[task]} & {label} & {b['mixed']:.5f} & {b['rank4']:.5f} & {b['rank64']:.5f} & [{min(loo):.5f}, {max(loo):.5f}]")
table('e3_reference','Sensitivity of squared bias proxies to the full-data reference: mixed (3 rank-4 + 5 rank-64 runs), rank-4 only, rank-64 only, and the range over eight leave-one-reference-out ensembles. Ranges are sensitivity checks, not confidence intervals.','llrrrr','Task & Method & Mixed & Rank 4 & Rank 64 & Leave-one-out range',rows)
main=[]; full=[]; geo=[]; rand=[]
for task,o in E5.items():
    e,h=o['euclidean_then_fisher'],o['h_projection'];N=NAMES[task]
    main.append(f"{N} & {100*e['oracle']:.2f}\\% & {100*e['snip']:.2f}\\% & {100*o['random_expected']:.4f}\\% & ${o['enrichment']:.2f}\\times$ \\\\")
    full.append(f"{N} & {o['K']} & {e['oracle']:.4f} & {e['snip']:.4f} & {o['random_expected']:.6f} & ${o['enrichment']:.2f}\\times$")
    geo.append(f"{N} & {100*e['snip']:.3f}\\% & {100*h['snip']:.3f}\\% & {100*o['h_joint_recovery']:.3f}\\% & {h['energy']/e['energy']:.3f}")
    rand.append(f"{N} & {100*o['random_expected']:.4f} & {100*o['random_mean']:.4f} & [{100*o['random_q025']:.4f}, {100*o['random_q975']:.4f}] & {o['random_exceedances']}")
(P/'e5_main_rows.tex').write_text('\n'.join(main)+'\n')
table('e5_recoverability','E5 coordinate-energy diagnostics on an explicitly full-data LoRA target proxy. $q_E=(I-PP^\\top)\\Delta\\theta$; oracle and SNIP capture fractions of $\\sum_i\\hat H_{ii}q_{E,i}^2$. The uniform-coordinate reference is $K/D$, $D=1{,}769{,}472$.','lrrrrr','Task & $K$ & Oracle & SNIP & $K/D$ & Enrichment',full)
table('e5_geometry','Effect of projection geometry with the same target, Fisher diagonal, and saved SNIP support. Coordinate capture on $q_E$ and $q_H$ uses each residual\'s own total weighted energy. Joint recovery refits the compressed and sparse coefficients under the diagonal metric and is normalized by $\\|q_H\\|_{\\hat H}^2$.','lrrrr','Task & Capture on $q_E$ & Capture on $q_H$ & Joint recovery & $\\|q_H\\|_{\\hat H}^2/\\|q_E\\|_{\\hat H}^2$',geo,'scriptsize')
table('e5_random','Uniform random-support check: 10,000 independent size-$K$ subsets drawn without replacement from all $D$ original coordinates (values in percent). The central 95\\% range describes random-support variation; it is not an interval for the mean. The last column counts draws reaching the observed SNIP capture on $q_E$.','lrrrr','Task & Exact mean & Sample mean & Central 95\\% range & Exceedances',rand)
