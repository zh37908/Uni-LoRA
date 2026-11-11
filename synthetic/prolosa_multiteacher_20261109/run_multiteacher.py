"""Frozen-config, independent-teacher follow-up to the spike/slab experiment.

The old run_search module supplies Adapter and train unchanged.  Its prepared
bundle supplies only frozen adapter initialization assets; no old teacher data
is used for a new teacher.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = HERE.parent / "prolosa_spike_slab_teacher_20260925"
PROTOCOL = REPO / "ICLR_2027/revision/reviewer_five_questions_20261109/q4_multiteacher_protocol_zh.txt"
sys.path.insert(0, str(OLD))
import run_search as original

sys.path.insert(0, str(HERE))
import teacher as teacher_source


core = original.core
sha = core.sha
atomic = core.atomic
base = core.base
METHODS = ("LoRA", "Uni-LoRA", "ProLoSA")
NOISE = (0.0, 0.5, 1.0, 2.0, 4.0)
TEACHER_SEEDS = tuple(range(100, 110))
REPEATS = tuple(range(20))
LONG_CONTROL = True  # Chosen before seeing any new teacher outcome.


def files():
    paths = {
        "runner": Path(__file__),
        "teacher_constructor": HERE / "teacher.py",
        "training_source": OLD / "run_search.py",
        "old_selection": OLD / "selection.json",
        "old_test_inputs": OLD / "test.pt",
        "old_prepared": OLD / "prepared.pt",
        "old_prepared_manifest": OLD / "prepared.json",
        "protocol": PROTOCOL,
    }
    paths.update({"core_" + key: path for key, path in core.files().items()})
    return paths


def file_hashes():
    return {key: sha(path) for key, path in files().items()}


def noise_tag(value):
    return "l" + str(value).replace(".0", "").replace(".", "p")


def check_config(configs):
    expected = {
        "LoRA": (0, 0, 0.0002, 1.0),
        "Uni-LoRA": (576, 0, 0.0005, 1.0),
        "ProLoSA": (512, 64, 0.0005, 4.0),
    }
    assert set(configs) == set(expected)
    for method, (bank, sparse, lr, multiplier) in expected.items():
        cfg = configs[method]
        assert cfg["method"] == method
        assert (cfg["bank"], cfg["sparse"], cfg["lr"], cfg["residual_lr_multiplier"]) == (
            bank, sparse, lr, multiplier
        )
        assert cfg["score_steps"] == 128
        if method == "ProLoSA":
            assert cfg["warmup"] == 256


def atomic_torch(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_numpy(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def init_assets():
    _, old_bundle = original.require()
    # Never pass the seed-13 teacher from the old prepared bundle to a fit.
    return {"matched": old_bundle["matched"], "banks": old_bundle["banks"]}


def build_grid():
    # Teacher is the fastest-moving index: shard i of ten owns teacher 100+i.
    grid = []
    for noise in NOISE:
        for method in METHODS:
            for repeat in REPEATS:
                for teacher_id, seed in enumerate(TEACHER_SEEDS):
                    grid.append(
                        dict(
                            teacher_id=teacher_id,
                            teacher_seed=seed,
                            repeat=repeat,
                            noise_multiple=noise,
                            method=method,
                            long_control=LONG_CONTROL and noise in (1.0, 4.0) and repeat < 5,
                            out=f"fits/t{seed}/{method}/{noise_tag(noise)}/r{repeat:02d}",
                        )
                    )
    assert len(grid) == 3000
    return grid


def prepare():
    if (HERE / "spec.json").exists():
        spec = require_spec()
        print("Already prepared", sha(HERE / "spec.json"), len(spec["grid"]), flush=True)
        return
    selected = json.loads((OLD / "selection.json").read_text())
    configs = selected["selected_configs"]
    check_config(configs)
    assert selected["selected"] == {"LoRA": "c000", "Uni-LoRA": "c004", "ProLoSA": "c077"}
    init_assets()  # Enforce the old source and initializer audit before generation.
    model = core.model_load()
    # The original locked test.pt contains the exact seed-1000050 inputs.
    # Reuse only its inputs and calculate fresh truth for every new teacher.
    assert selected["test_sha256"] == sha(OLD / "test.pt")
    x_test = torch.load(OLD / "test.pt", map_location="cpu", weights_only=False)["x"].cuda()
    assert x_test.shape == (1024, 32, 64)
    test_id = base.digest(x_test.cpu().numpy())
    assert test_id == selected["test_id"]
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_torch(HERE / "test.pt", x_test.cpu())
    teacher_manifests = {}
    for teacher_id, seed in enumerate(TEACHER_SEEDS):
        folder = HERE / "teachers" / f"t{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            assert m["teacher_id"] == teacher_id and m["teacher_seed"] == seed
            assert m["teacher_source_sha256"] == sha(HERE / "teacher.py")
            for key, filename in (("bundle_sha256", "teacher.pt"), ("truth_sha256", "truth.npy"), ("diagnostics_sha256", "diagnostics.json")):
                assert m[key] == sha(folder / filename)
            assert m["test_id"] == test_id
        else:
            tensors, info = teacher_source.construct(core, model, seed)
            teacher = {key: value.cuda() for key, value in tensors["teacher"].items()}
            truth = base.predict(model, x_test, teacher).cpu().numpy()
            assert truth.shape == (1024, 16) and np.isfinite(truth).all()
            atomic_torch(folder / "teacher.pt", tensors)
            atomic_numpy(folder / "truth.npy", truth)
            atomic(folder / "diagnostics.json", info)
            m = dict(
                teacher_id=teacher_id, teacher_seed=seed,
                teacher_hash=core.delta_digest(teacher),
                teacher_source_sha256=sha(HERE / "teacher.py"),
                bundle_sha256=sha(folder / "teacher.pt"),
                truth_sha256=sha(folder / "truth.npy"),
                diagnostics_sha256=sha(folder / "diagnostics.json"),
                test_id=test_id, noise_unit=info["noise_unit"],
                displacement=info["displacement"], scale=info["scale"],
            )
            atomic(manifest_path, m)
        teacher_manifests[str(seed)] = dict(m, manifest_sha256=sha(manifest_path))
        print("Prepared teacher", seed, "c", m["scale"], "noise unit", m["noise_unit"], flush=True)
    spec = dict(
        title="Q4 independent spike/slab Transformer teachers",
        teacher_seeds=list(TEACHER_SEEDS), noise_multiples=list(NOISE),
        methods=list(METHODS), repeats=len(REPEATS), train_n=256,
        updates=1000, long_control=LONG_CONTROL, long_updates=3000,
        selected_ids=selected["selected"], configs=configs,
        dependencies=file_hashes(), selection_source_sha256=sha(OLD / "selection.json"),
        test_sha256=sha(HERE / "test.pt"), test_id=test_id,
        teacher_manifests=teacher_manifests, grid=build_grid(),
    )
    atomic(HERE / "spec.json", spec)
    print("Prepared experiment", len(spec["grid"]), "fits", flush=True)


def require_spec():
    spec = json.loads((HERE / "spec.json").read_text())
    assert spec["dependencies"] == file_hashes(), "Source/dependency changed after preparation"
    assert spec["test_sha256"] == sha(HERE / "test.pt")
    assert spec["grid"] == build_grid()
    assert spec["teacher_seeds"] == list(TEACHER_SEEDS)
    assert spec["noise_multiples"] == list(NOISE)
    assert spec["methods"] == list(METHODS)
    assert spec["repeats"] == 20 and spec["updates"] == 1000
    check_config(spec["configs"])
    for seed in TEACHER_SEEDS:
        folder = HERE / "teachers" / f"t{seed}"
        m = json.loads((folder / "manifest.json").read_text())
        assert spec["teacher_manifests"][str(seed)] == dict(m, manifest_sha256=sha(folder / "manifest.json"))
        for key, filename in (("bundle_sha256", "teacher.pt"), ("truth_sha256", "truth.npy"), ("diagnostics_sha256", "diagnostics.json")):
            assert m[key] == sha(folder / filename)
    return spec


def load_teacher(seed):
    folder = HERE / "teachers" / f"t{seed}"
    tensors = torch.load(folder / "teacher.pt", map_location="cpu", weights_only=False)
    truth = np.load(folder / "truth.npy").astype(float)
    m = json.loads((folder / "manifest.json").read_text())
    teacher = {key: value.cuda() for key, value in tensors["teacher"].items()}
    assert core.delta_digest(teacher) == m["teacher_hash"]
    return teacher, truth, m


def training_pair(teacher_id, repeat):
    x_seed = 2000000 + 1000 * teacher_id + repeat
    noise_seed = 3000000 + 1000 * teacher_id + repeat
    x = torch.randn(256, 32, 64, generator=torch.Generator(device="cuda").manual_seed(x_seed), device="cuda")
    noise = torch.randn(256, 16, generator=torch.Generator(device="cuda").manual_seed(noise_seed), device="cuda")
    return x, noise, x_seed, noise_seed


def smoke():
    spec = require_spec()
    path = HERE / "smoke.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old["spec_sha256"] == sha(HERE / "spec.json") and old["source_sha256"] == sha(__file__)
        assert set(old["methods"]) == set(METHODS)
        print("Smoke already passed", flush=True)
        return
    init = init_assets()
    model = core.model_load()
    teacher, truth, _ = load_teacher(100)
    x_test = torch.load(HERE / "test.pt", map_location="cpu", weights_only=False).cuda()[:32]
    x = torch.randn(256, 32, 64, generator=torch.Generator(device="cuda").manual_seed(9000000), device="cuda")
    noise = torch.randn(256, 16, generator=torch.Generator(device="cuda").manual_seed(9000001), device="cuda")
    clean = base.predict(model, x, teacher)
    y = clean + spec["teacher_manifests"]["100"]["noise_unit"] * noise
    a_full = original.Adapter(spec["configs"]["LoRA"], init)
    a_uni = original.Adapter(spec["configs"]["Uni-LoRA"], init)
    torch.testing.assert_close(a_full.vector(), a_uni.vector(), rtol=0, atol=0)
    report = {}
    for method in METHODS:
        adapter = original.Adapter(spec["configs"][method], init)
        pred, _, rows = original.train(model, adapter, x, y, clean, x_test, truth[:32], [400])
        assert pred["step400"].shape == (32, 16)
        assert math.isfinite(rows[0]["risk"])
        assert int(adapter.mask.sum()) == (64 if method == "ProLoSA" else 0)
        report[method] = dict(seconds=rows[0]["seconds"], risk=rows[0]["risk"], support=int(adapter.mask.sum()))
        print("Smoke", method, report[method], flush=True)
    atomic(path, dict(spec_sha256=sha(HERE / "spec.json"), source_sha256=sha(__file__), methods=report, job=os.environ.get("SLURM_JOB_ID")))


def check_completed(result_path, spec, config, teacher_manifest):
    result = json.loads(result_path.read_text())
    assert result["spec"] == spec and result["config"] == config
    assert result["spec_sha256"] == sha(HERE / "spec.json")
    assert result["source_sha256"] == sha(__file__)
    assert result["train_source_sha256"] == sha(OLD / "run_search.py")
    assert result["teacher_source_sha256"] == sha(HERE / "teacher.py")
    assert result["teacher_hash"] == teacher_manifest["teacher_hash"]
    assert result["teacher_bundle_sha256"] == teacher_manifest["bundle_sha256"]
    assert [row["step"] for row in result["rows"]] == ([1000, 3000] if spec["long_control"] else [1000])
    folder = result_path.parent
    for artifact in ("predictions", "factors", "support"):
        assert result[artifact + "_sha256"] == sha(folder / (artifact + ".npz"))
    return result


def run(shard, shards):
    spec = require_spec()
    assert 0 <= shard < shards and shards == 5
    smoke_report = json.loads((HERE / "smoke.json").read_text())
    assert smoke_report["spec_sha256"] == sha(HERE / "spec.json")
    assert smoke_report["source_sha256"] == sha(__file__)
    init = init_assets()
    model = core.model_load()
    x_test = torch.load(HERE / "test.pt", map_location="cpu", weights_only=False).cuda()
    assert base.digest(x_test.cpu().numpy()) == spec["test_id"]
    owned = spec["grid"][shard::shards]
    expected_seeds = {TEACHER_SEEDS[shard], TEACHER_SEEDS[shard + shards]}
    assert len(owned) == 600 and {entry["teacher_seed"] for entry in owned} == expected_seeds
    states = {seed: load_teacher(seed) for seed in expected_seeds}
    pair_cache = {}
    for i, entry in enumerate(owned):
        seed = entry["teacher_seed"]
        teacher, truth, manifest = states[seed]
        config = spec["configs"][entry["method"]]
        out = HERE / entry["out"]
        out.mkdir(parents=True, exist_ok=True)
        result_path = out / "result.json"
        if result_path.exists():
            check_completed(result_path, entry, config, manifest)
            continue
        repeat = entry["repeat"]
        pair_key = (entry["teacher_id"], repeat)
        if pair_key not in pair_cache:
            pair_cache[pair_key] = training_pair(entry["teacher_id"], repeat)
        x, noise, x_seed, noise_seed = pair_cache[pair_key]
        clean = base.predict(model, x, teacher)
        sigma = entry["noise_multiple"] * manifest["noise_unit"]
        y = clean + sigma * noise
        adapter = original.Adapter(config, init)
        checkpoints = [1000, 3000] if entry["long_control"] else [1000]
        preds, factors, rows = original.train(model, adapter, x, y, clean, x_test, truth, checkpoints)
        np.savez_compressed(out / "predictions.npz", **preds)
        np.savez_compressed(out / "factors.npz", **factors)
        support = dict(mask=adapter.mask.cpu().numpy(), scores=adapter.scores.cpu().numpy())
        if hasattr(adapter, "bank"):
            support.update(bank=adapter.bank.detach().cpu().numpy(), residual=adapter.residual.detach().cpu().numpy())
        np.savez_compressed(out / "support.npz", **support)
        assert int(adapter.mask.sum()) == config["sparse"]
        result = dict(
            spec=entry, config=config, rows=rows, sigma=sigma,
            teacher_id=entry["teacher_id"], teacher_seed=seed,
            teacher_hash=manifest["teacher_hash"], teacher_bundle_sha256=manifest["bundle_sha256"],
            teacher_source_sha256=sha(HERE / "teacher.py"),
            source_sha256=sha(__file__), train_source_sha256=sha(OLD / "run_search.py"),
            spec_sha256=sha(HERE / "spec.json"),
            selection_sha256=spec["selection_source_sha256"],
            test_id=spec["test_id"], test_sha256=spec["test_sha256"],
            truth_sha256=manifest["truth_sha256"],
            train_x_id=base.digest(x.cpu().numpy()), noise_id=base.digest(noise.cpu().numpy()),
            train_y_id=base.digest(y.cpu().numpy()),
            x_seed=x_seed, noise_seed=noise_seed, minibatch_seed=81000,
            effective_parameters=4608 if entry["method"] == "LoRA" else config["bank"] + config["sparse"],
            allocated_parameter_elements=sum(p.numel() for p in adapter.parameters()),
            steps=max(checkpoints), gpu=torch.cuda.get_device_name(0),
            torch_version=torch.__version__, job=os.environ.get("SLURM_JOB_ID"),
        )
        for artifact in ("predictions", "factors", "support"):
            result[artifact + "_sha256"] = sha(out / (artifact + ".npz"))
        atomic(result_path, result)
        print("Fit", seed, i + 1, "/", len(owned), entry["method"], entry["noise_multiple"], repeat, rows, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "status"))
    parser.add_argument("--shards", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "status":
        spec = require_spec()
        counts = {str(seed): 0 for seed in TEACHER_SEEDS}
        for entry in spec["grid"]:
            if (HERE / entry["out"] / "result.json").exists():
                counts[str(entry["teacher_seed"])] += 1
        print(json.dumps(dict(completed=sum(counts.values()), total=len(spec["grid"]), by_teacher=counts), indent=2))
        return
    core.old.setup()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "smoke":
        smoke()
    else:
        run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")), args.shards)


if __name__ == "__main__":
    main()
