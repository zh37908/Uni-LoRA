"""Fixed-teacher predictive bias/variance experiments; all predictions retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import model_frozen as base

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent / 'results_revision5_fixed_teacher/teachers.pt'
GRID = [dict(sigma=s, dim=d, repeat=r, optimizer_seed=81000)
        for s in [0., .25, .5, 1.]
        for d in ([128, 8192] if s in [0., .5] else [32, 128, 512, 8192])
        for r in range(20)]
CONTROLS = [dict(sigma=1., dim=d, repeat=r, optimizer_seed=o)
            for d in [128, 8192] for r in range(3) for o in [81001, 81002]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare():
    payload = torch.load(TEACHER, map_location='cpu', weights_only=False)
    perm = np.random.default_rng(913).permutation(8192)
    assert np.array_equal(perm % 128, payload['matched_assignment'].numpy())
    assert np.array_equal(perm % 64, payload['assignment'].numpy())
    # Common permutation, successively subdivided buckets: exact nested spaces.
    for small, large in [(32, 128), (128, 512)]:
        assert np.array_equal((perm % large) % small, perm % small)
    np.savez_compressed(HERE / 'projections.npz', **{f'd{d}': perm % d for d in [32,128,512]})
    np.save(HERE / 'clean_test_targets.npy', payload['profiles']['concentrated']['y_test'].numpy())
    manifest = dict(protocol='fixed_teacher_predictive_bv_v1',
        teacher_path=str(TEACHER), teachers_sha256=sha(TEACHER),
        target_id=payload['profiles']['concentrated']['target_id'],
        test_id=base.digest(payload['x_test'].numpy()),
        source_sha256=sha(__file__), model_sha256=sha(base.__file__),
        projections_sha256=sha(HERE / 'projections.npz'),
        truth_sha256=sha(HERE / 'clean_test_targets.npy'),
        n=256, test_n=1024, outputs=16, repeats=20, steps=1000,
        optimizer='AdamW', learning_rate=.005, weight_decay=0., clip_norm=1.,
        batch_size=128, optimizer_seed=81000, initialization='zero update',
        noise_units='absolute output-coordinate standard deviation',
        displacement=payload['profiles']['concentrated']['displacement'],
        base_output_variance=payload['output_variance'],
        main_grid=GRID, optimizer_controls=CONTROLS,
        long_control='sigma in {0,1}, dim in {128,8192}, repeats 0..4: extend to 3000 steps',
        interpretation='direct Q/V weight updates, not standard LoRA factors')
    if (HERE / 'manifest.json').exists():
        assert json.loads((HERE / 'manifest.json').read_text()) == manifest
    else:
        atomic(HERE / 'manifest.json', manifest)
    print('Prepared', len(GRID), 'main runs and', len(CONTROLS), 'optimizer controls', flush=True)


class Adapter(nn.Module):
    def __init__(self, dim, assignment):
        super().__init__()
        self.z = nn.Parameter(torch.zeros(dim, device='cuda'))
        self.register_buffer('assignment', assignment)
        self.scale = (8192 / dim) ** .5

    def forward(self):
        return self.z if self.assignment is None else self.z[self.assignment] / self.scale


def run(index, mode):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['source_sha256'] == sha(__file__)
    assert manifest['model_sha256'] == sha(base.__file__)
    assert manifest['teachers_sha256'] == sha(TEACHER)
    assert manifest['projections_sha256'] == sha(HERE / 'projections.npz')
    spec = (CONTROLS if mode == 'optimizer' else GRID)[index]
    folder = HERE / mode / f'{index:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        result = json.loads((folder / 'result.json').read_text())
        assert result['source_sha256'] == sha(__file__) and result['spec'] == spec
        assert result['predictions_sha256'] == sha(folder / 'predictions.npz')
        return
    torch.manual_seed(6000)
    torch.cuda.manual_seed_all(6000)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(TEACHER, map_location='cpu', weights_only=False)
    model = base.Block().cuda()
    model.load_state_dict(payload['state'])
    model.eval().requires_grad_(False)
    frozen = payload['profiles']['concentrated']
    theta = frozen['theta'].cuda()
    xtest = payload['x_test'].cuda()
    truth = frozen['y_test'].numpy().astype(np.float64)
    assert base.digest(frozen['theta'].double().numpy()) == manifest['target_id']
    teacher_error = float(np.mean((base.predict(model, xtest, theta).cpu().numpy() - truth)**2))
    assert teacher_error < 1e-10
    # Independent datasets across repeats; paired X/epsilon across dimensions/noise.
    data_seed = 920000 + spec['repeat']
    generator = torch.Generator(device='cuda').manual_seed(data_seed)
    x = torch.randn(256, 32, 64, generator=generator, device='cuda')
    noise = torch.randn(256, 16, generator=generator, device='cuda')
    clean = base.predict(model, x, theta)
    y = clean + spec['sigma'] * noise
    assignment = None if spec['dim'] == 8192 else torch.from_numpy(
        np.load(HERE / 'projections.npz')[f'd{spec["dim"]}']).cuda()
    adapter = Adapter(spec['dim'], assignment)
    assert sum(p.numel() for p in adapter.parameters()) == spec['dim']
    assert torch.count_nonzero(adapter()) == 0
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.005, weight_decay=0.)
    # Identical optimization randomness for every primary fit and every repeat.
    batches = torch.Generator(device='cuda').manual_seed(spec['optimizer_seed'])
    checkpoints = [1000]
    if mode == 'main' and spec['repeat'] < 5 and spec['sigma'] in [0.,1.] and spec['dim'] in [128,8192]:
        checkpoints += [3000]
    if mode == 'smoke':
        checkpoints = [1000]
    predictions, updates, rows = {}, {}, []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, max(checkpoints)+1):
        indices = torch.randint(256, (128,), generator=batches, device='cuda')
        loss = F.mse_loss(model(x[indices], adapter()), y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        optimizer.step()
        if step in checkpoints:
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            delta = adapter().detach()
            pred = base.predict(model, xtest, delta).cpu().numpy()
            train = base.predict(model, x, delta)
            assert np.isfinite(pred).all() and torch.isfinite(delta).all()
            predictions[f'step{step}'] = pred
            updates[f'step{step}'] = delta.cpu().numpy()
            row = dict(step=step, mse=float(np.mean((pred.astype(np.float64)-truth)**2)),
                       train_mse=float(F.mse_loss(train,y)), clean_train_mse=float(F.mse_loss(train,clean)), seconds=seconds)
            rows.append(row)
            print(mode, index, spec, row, flush=True)
    np.savez_compressed(folder / 'predictions.npz', **predictions)
    np.savez_compressed(folder / 'updates.npz', **updates)
    atomic(folder / 'result.json', dict(spec=spec, mode=mode, rows=rows,
        source_sha256=sha(__file__), manifest_sha256=sha(HERE/'manifest.json'),
        predictions_sha256=sha(folder/'predictions.npz'), updates_sha256=sha(folder/'updates.npz'),
        target_id=manifest['target_id'], test_id=manifest['test_id'],
        train_x_id=base.digest(x.cpu().numpy()), noise_id=base.digest(noise.cpu().numpy()),
        train_y_id=base.digest(y.cpu().numpy()), data_seed=data_seed, teacher_test_mse=teacher_error,
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','smoke','main','optimizer'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'prepare':
        prepare()
    else:
        assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'Use a Slurm GPU allocation'
        indices = [args.index] if args.mode == 'smoke' else range(args.index, len(CONTROLS if args.mode == 'optimizer' else GRID), args.shards)
        for index in indices:
            run(index, args.mode)
