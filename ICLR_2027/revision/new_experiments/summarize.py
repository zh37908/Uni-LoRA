"""Strict, complete-grid summaries; never pool predictions across teachers."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
METHODS = ['lora', 'compressed', 'random', 'gradient', 'snip', 'energy',
           'snip_independent', 'snip_same_half']


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    assert len(x) in (5, 20) and np.isfinite(x).all()
    t = {5: 2.7764451051977987, 20: 2.093024054408263}[len(x)]
    sd = float(x.std(ddof=1))
    mean = float(x.mean())
    half = t * sd / math.sqrt(len(x))
    return dict(mean=mean, sd=sd, ci_low=mean-half, ci_high=mean+half)


def fixed_decomposition(predictions, truth, target_ids, expected_target_id):
    assert set(target_ids) == {expected_target_id}, 'Mixed teachers are forbidden'
    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    assert pred.ndim == truth.ndim+1 and pred.shape[1:] == truth.shape
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    center = pred.mean(axis=0)
    bias = float(np.mean((center-truth)**2))
    variance = float(np.mean((pred-center)**2))
    risks = ((pred-truth)**2).reshape(len(pred), -1).mean(axis=1)
    assert np.isclose(risks.mean(), bias+variance, atol=1e-12, rtol=1e-10)
    return bias, variance, risks


def synthetic():
    import torch
    root = ROOT/'synthetic/results_revision5_fixed_teacher'
    teachers = torch.load(root/'teachers.pt', map_location='cpu', weights_only=False)
    grid = [(p, n, noise, seed) for p in ['concentrated', 'diffuse']
            for n in [64, 256, 1024, 4096] for noise in [.25, .5, 1.]
            for seed in range(20)]
    groups, raw, hashes = {}, [], set()
    for i, key in enumerate(grid):
        directory = root/'trials_v2'/f'{i:04d}'
        result = json.loads((directory/'result.json').read_text())
        profile, n, noise, seed = key
        assert (result['profile'], result['n'], result['noise_std'], result['seed']) == key
        assert not result['smoke'] and result['protocol'] == 'pilot_reset_matched_half_v2'
        assert result['target_id'] == teachers['profiles'][profile]['target_id']
        assert result['test_id'] == digest(teachers['x_test'].numpy())
        hashes.add(result['source_sha256'])
        assert [r['method'] for r in result['rows']] == METHODS
        with np.load(directory/'predictions.npz') as arrays:
            for row in result['rows']:
                method = row['method']
                assert row['params'] == (1024 if method == 'lora' else 128)
                assert row['charged_gradient_steps'] == 1000 and row['available_examples'] == n
                half_fit = method in ['snip_independent', 'snip_same_half']
                assert row['fitting_examples'] == (n//2 if half_fit else n)
                assert row['total_unique_examples'] == (n//2 if method == 'snip_same_half' else n)
                assert row['selection_examples'] == (0 if method in METHODS[:2] else n//2 if half_fit else n)
                assert len(set(row['support'])) == (0 if method in METHODS[:2] else 64)
                groups.setdefault((profile, n, noise, method), []).append(
                    (seed, result['target_id'], arrays[method].copy(), row))
                raw.append(dict(profile=profile, n=n, noise_std=noise, seed=seed,
                                target_id=result['target_id'],
                                **{k: v for k, v in row.items() if k != 'support'}))
    assert len(hashes) == 1, 'Mixed source versions require an explicit audit'
    rows, paired, risk_arrays = [], [], {}
    for key, runs in sorted(groups.items()):
        profile, n, noise, method = key
        runs.sort(key=lambda r: r[0])
        assert [r[0] for r in runs] == list(range(20))
        teacher = teachers['profiles'][profile]
        assert digest(teacher['theta'].double().numpy()) == teacher['target_id']
        bias, var, risks = fixed_decomposition(
            [r[2] for r in runs], teacher['y_test'].numpy(),
            [r[1] for r in runs], teacher['target_id'])
        assert np.allclose(risks, [r[3]['mse'] for r in runs], atol=1e-12, rtol=1e-10)
        risk_arrays[key] = risks
        captures = [r[3]['residual_capture'] for r in runs]
        row = dict(profile=profile, n=n, noise_std=noise, method=method,
                   target_id=teacher['target_id'], seeds=20, mse=float(risks.mean()),
                   bias_squared=bias, variance=var,
                   residual_capture_mean=None if captures[0] is None else float(np.mean(captures)),
                   params=runs[0][3]['params'],
                   selection_examples=runs[0][3]['selection_examples'],
                   fitting_examples=runs[0][3]['fitting_examples'],
                   total_unique_examples=runs[0][3]['total_unique_examples'],
                   charged_seconds_mean=float(np.mean([r[3]['charged_seconds'] for r in runs])))
        row.update({f'mse_{k}': v for k, v in mean_ci(risks).items() if k != 'mean'})
        rows.append(row)
    for profile in ['concentrated', 'diffuse']:
        for n in [64, 256, 1024, 4096]:
            for noise in [.25, .5, 1.]:
                pairs = [('compressed', m) for m in METHODS if m != 'compressed']
                pairs += [('snip_same_half', 'snip_independent'), ('snip', 'snip_independent')]
                for base, method in pairs:
                    gain = risk_arrays[profile, n, noise, base]-risk_arrays[profile, n, noise, method]
                    paired.append(dict(profile=profile, n=n, noise_std=noise,
                                       baseline=base, method=method, **mean_ci(gain)))
    out = P/'summary_synthetic';out.mkdir(exist_ok=True)
    write_csv(out/'per_seed.csv', raw)
    write_csv(out/'conditional_summary.csv', rows)
    write_csv(out/'paired_risk_gains.csv', paired)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows, paired=paired,
        source_sha256=next(iter(hashes)), target_ids={p: v['target_id'] for p, v in teachers['profiles'].items()}), indent=2)+'\n')
    lines = ['# Fixed-teacher synthetic controls', '',
        'Complete: 480 trials, 8 methods, 20 seeds per teacher/sample-size/noise cell.', '',
        'Bias² and variance are empirical finite-ensemble quantities (variance divisor 20). '
        'Their sum equals the mean clean-test MSE, without a factor 1/2. '
        'The ensemble varies training data, noise and optimization seed jointly; '
        'it does not separately identify statistical and optimization variance. '
        'Each teacher is analyzed separately; no cross-teacher decomposition is computed.', '',
        'Pointwise paired t intervals quantify seed uncertainty conditional on the fixed teacher, '
        'projection and test inputs. They do not cover teacher or test-distribution uncertainty. '
        'Positive paired gain means the method reduces risk relative to the named baseline.', '',
        'The energy support is an oracle for Euclidean residual coordinate energy. '
        'Residual capture is a coordinate proxy, not a nonlinear bias-recovery theorem. '
        'All diagnostic hybrids reset after pilot selection. LoRA has 1,024 parameters; '
        'all other methods have 128. The LoRA comparison is a budget tradeoff.', '',
        'Compare snip_same_half with snip_independent to hold selection and fitting sizes '
        'fixed at n/2: their unique consumed data are n/2 and n. Compare snip with '
        'snip_independent to hold total unique data at n: fitting sizes then differ. '
        'Report both contrasts; neither alone isolates every effect of data reuse.', '',
        '| Teacher | n | Noise | Method | MSE | Bias² | Variance |',
        '|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['profile']} | {r['n']} | {r['noise_std']} | {r['method']} | {r['mse']:.6g} | {r['bias_squared']:.6g} | {r['variance']:.6g} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('Synthetic summary complete:', out)


def glue():
    locked = json.loads((P/'locked_selection.json').read_text())
    paths = json.loads((P/'final_manifest.json').read_text())
    assert len(paths) == 40
    groups, raw = {}, []
    for path in paths:
        spec = json.loads(Path(path).read_text())
        result = json.loads((P/'results'/spec['id']/'result.json').read_text())
        assert result['spec'] == spec and spec['phase'] == 'final'
        assert not result['heldout_is_smoke_surrogate']
        assert spec['config_id'] == locked[spec['task']+'/'+spec['method']]['config_id']
        groups.setdefault((spec['task'], spec['method']), []).append(result)
        raw.append(dict(task=spec['task'], method=spec['method'], seed=spec['seed'],
                        config_id=spec['config_id'], score=result['heldout_score'],
                        loss=result['heldout_loss'], effective_adapter=result['effective_adapter_coordinates'],
                        allocated_adapter=result['trainable_adapter'], trainable_total=result['trainable_total'],
                        elapsed_seconds=result['elapsed_seconds'], peak_gpu_bytes=result['peak_gpu_bytes']))
    assert len(groups) == 8
    summary, paired = [], []
    for (task, method), runs in sorted(groups.items()):
        runs.sort(key=lambda r: r['spec']['seed'])
        assert [r['spec']['seed'] for r in runs] == list(range(201, 206))
        summary.append(dict(task=task, method=method, seeds=5,
                            effective_adapter=runs[0]['effective_adapter_coordinates'],
                            allocated_adapter=runs[0]['trainable_adapter'],
                            **{f'score_{k}': v for k, v in mean_ci([r['heldout_score'] for r in runs]).items()},
                            **{f'loss_{k}': v for k, v in mean_ci([r['heldout_loss'] for r in runs]).items()}))
    for task in ['cola', 'mrpc']:
        for baseline, method in [('lora', 'unilora'), ('unilora', 'prolosa'), ('lora', 'rosa'), ('rosa', 'prolosa')]:
            a, b = groups[task, baseline], groups[task, method]
            for metric in ['score', 'loss']:
                delta = [y['heldout_'+metric]-x['heldout_'+metric] for x, y in zip(a, b)]
                paired.append(dict(task=task, baseline=baseline, method=method, metric=metric,
                                   direction='method minus baseline', **mean_ci(delta)))
    tuning_seconds = 0.
    for path in json.loads((P/'tune_manifest.json').read_text()):
        spec = json.loads(Path(path).read_text())
        result = json.loads((P/'results'/spec['id']/'result.json').read_text())
        assert result['spec'] == spec and 'heldout_score' not in result
        tuning_seconds += result['elapsed_seconds']
    out = P/'summary_glue';out.mkdir(exist_ok=True)
    write_csv(out/'per_seed.csv', raw);write_csv(out/'summary.csv', summary)
    write_csv(out/'paired_differences.csv', paired)
    (out/'summary.json').write_text(json.dumps(dict(summary=summary, paired=paired,
        tuning_gpu_hours=tuning_seconds/3600,
        final_gpu_hours=sum(r['elapsed_seconds'] for r in raw)/3600), indent=2)+'\n')
    lines = ['# Fairly tuned GLUE controls', '',
        'Complete: 192 tuning runs on a train-only holdout; 40 locked final runs. '
        'Original development labels enter only final scoring. The same train split '
        'and five training seeds are paired across methods. Pointwise t intervals '
        'describe training-seed variation conditional on this split and evaluation set.', '',
        'LoRA r4 and the local RoSA r2 implementation each use 1,769,472 effective '
        'adapter coordinates. Uni-LoRA and ProLoSA each use 23,040. Comparisons '
        'across these two tiers describe a budget tradeoff. ProLoSA stores a dense '
        'masked sparse vector, so allocated tensor entries exceed effective coordinates. '
        'The separately shared classification head is included in trainable_total.', '',
        f'Tuning GPU hours: {tuning_seconds/3600:.2f}. '
        f'Final GPU hours: {sum(r["elapsed_seconds"] for r in raw)/3600:.2f}. '
        'These sum per-run elapsed training/evaluation time, excluding queue and imports.', '',
        '| Task | Method | Effective adapter | Score mean | Seed SD | 95% t interval |',
        '|---|---|---:|---:|---:|---|']
    for r in summary:
        lines.append(f"| {r['task']} | {r['method']} | {r['effective_adapter']} | {r['score_mean']:.5f} | {r['score_sd']:.5f} | [{r['score_ci_low']:.5f}, {r['score_ci_high']:.5f}] |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('GLUE summary complete:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode', choices=['glue', 'synthetic'])
    args = parser.parse_args()
    globals()[args.mode]()
