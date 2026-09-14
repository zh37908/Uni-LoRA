"""Persistent, locked multi-GPU experiment queue and validation-only selection.

Screen: 24 candidates/group, 5 epochs, seed 101.
Full validation: top 3/group, 20 epochs, seed 101.
Confirmation: top 2/group, 20 epochs, seed 102.
Final: best two-seed mean/group, 20 epochs, seeds 42--46 (80 runs).
"""
import argparse
import collections
import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ["cifar100", "dtd", "cifar10", "oxfordpets", "stanfordcars", "eurosat", "resisc45", "fgvc"]
GROUPS = [(m,d) for d in DATASETS for m in ("base", "large")]
PYTHON = "/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python"
STATE = ROOT / "state.json"


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def signature(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


@contextmanager
def locked():
    with (ROOT / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(STATE.read_text())
        yield state
        atomic(STATE, state)


def add(state, stage, model, dataset, config, seed, epochs):
    spec = dict(stage=stage, model=model, dataset=dataset, seed=seed, epochs=epochs, **config)
    ident = f"{stage}_{model}_{dataset}_{signature(spec)[:12]}"
    if ident in state["tasks"]:
        return
    path = ROOT / "specs" / f"{ident}.json"
    atomic(path, spec)
    state["tasks"][ident] = dict(spec=spec, status="pending", attempts=[], path=str(path))


def configs():
    return [dict(vector_lr=lr, head_lr=head, ratio=ratio, sparse_lr_mult=0.2, support_warmup_ratio=0.1)
            for lr, head, ratio in itertools.product((0.002, 0.01, 0.05), (0.002, 0.01), (2,4,8,12))]


def configuration(task):
    return {k:task["spec"][k] for k in ("vector_lr", "head_lr", "ratio", "sparse_lr_mult", "support_warmup_ratio")}


def result(task):
    r = json.loads(Path(task["result"]).read_text())
    assert r["spec_sha256"] == signature(task["spec"])
    if task["spec"]["stage"] != "final":
        assert "test_accuracy" not in r
    return r


def ranking(task):
    r = result(task)
    return (-r["best_val_accuracy"], r["best_val_loss"], signature(configuration(task)))


def tasks_for(state, stage, group=None):
    return [v for v in state["tasks"].values() if v["spec"]["stage"] == stage and
            (group is None or (v["spec"]["model"],v["spec"]["dataset"]) == group)]


def complete(tasks):
    return bool(tasks) and all(t["status"] == "done" for t in tasks)


def advance(state):
    if not complete(tasks_for(state, "smoke")):
        return
    if not state.get("screen_initialized"):
        for model,dataset in GROUPS:
            for config in configs(): add(state, "screen", model, dataset, config, 101, 5)
        state["screen_initialized"] = True
    for group in GROUPS:
        model,dataset = group
        key = f"{model}/{dataset}"
        screen = tasks_for(state, "screen", group)
        if complete(screen) and not tasks_for(state, "validate", group):
            for t in sorted(screen, key=ranking)[:3]:
                add(state, "validate", model, dataset, configuration(t), 101, 20)
        full = tasks_for(state, "validate", group)
        if complete(full) and not tasks_for(state, "confirm", group):
            for t in sorted(full, key=ranking)[:2]:
                add(state, "confirm", model, dataset, configuration(t), 102, 20)
        second = tasks_for(state, "confirm", group)
        if complete(second) and key not in state["selected"]:
            paired = []
            for t in second:
                config = configuration(t)
                first = next(x for x in full if configuration(x) == config)
                scores = [result(first), result(t)]
                paired.append(dict(config=config,
                                   mean_val_accuracy=sum(x["best_val_accuracy"] for x in scores)/2,
                                   mean_val_loss=sum(x["best_val_loss"] for x in scores)/2,
                                   validation_results=[first["result"], t["result"]]))
            choice = min(paired, key=lambda x:(-x["mean_val_accuracy"],x["mean_val_loss"],signature(x["config"])))
            choice["selected_at"] = time.time()
            state["selected"][key] = choice
            atomic(ROOT / "selected_configs.json", state["selected"])
    # Global barrier: no final/test run until all 16 groups have validated settings.
    if len(state["selected"]) == len(GROUPS) and not state.get("final_initialized"):
        for (model,dataset) in GROUPS:
            cfg = state["selected"][f"{model}/{dataset}"]["config"]
            for seed in range(42,47): add(state, "final", model, dataset, cfg, seed, 20)
        assert len(tasks_for(state, "final")) == 80
        state["final_initialized"] = True
        state["final_started_at"] = time.time()
    if complete(tasks_for(state, "final")):
        state["status"] = "complete"
        summarize_final(state)


def summarize_final(state):
    import statistics
    summary = {}
    for group in GROUPS:
        ts = tasks_for(state, "final", group)
        rows = [result(t) for t in sorted(ts,key=lambda t:t["spec"]["seed"])]
        acc = [r["test_accuracy"] * 100 for r in rows]
        summary["/".join(group)] = dict(mean=statistics.mean(acc), std=statistics.stdev(acc),
                                        seed_accuracy=acc, config=state["selected"]["/".join(group)]["config"])
    atomic(ROOT / "final_summary.json", summary)


def status(state):
    counts = {}
    for stage in ("smoke", "screen", "validate", "confirm", "final"):
        counts[stage] = dict(collections.Counter(t["status"] for t in tasks_for(state, stage)))
    return dict(status=state["status"], counts=counts, selected_groups=len(state["selected"]),
                updated_at=time.time(), workers=state.get("workers",{}))


def initialize():
    if STATE.exists():
        print("Existing queue retained:", STATE)
        return
    state = dict(status="running", created_at=time.time(), tasks={}, selected={}, workers={})
    cfg = dict(vector_lr=0.01, head_lr=0.01, ratio=4, sparse_lr_mult=0.2, support_warmup_ratio=0.1)
    for dataset in ("cifar100", "dtd"):
        for model in ("base", "large"): add(state, "smoke", model, dataset, cfg, 101, 2)
    atomic(STATE, state)
    atomic(ROOT / "search_protocol.json", dict(
        created_at=state["created_at"], group_count=16, configs=configs(),
        stages=[dict(name="screen", epochs=5, seed=101, candidates=24),
                dict(name="validate", epochs=20, seed=101, top=3),
                dict(name="confirm", epochs=20, seed=102, top=2)],
        formal_seeds=list(range(42,47)), formal_runs=80,
        selection="highest two-seed mean best validation accuracy; loss then config hash break ties",
        test_policy="test split is never opened during smoke/screen/validate/confirm",
        scope="best among screened candidate configurations; not a global optimum guarantee"))


def worker(worker_id):
    key = f'{os.environ.get("SLURM_JOB_ID","local")}/{worker_id}'
    deadline = time.time() + 70*3600
    priorities = {s:i for i,s in enumerate(("smoke", "confirm", "validate", "screen", "final"))}
    while time.time() < deadline:
        selected = None
        with locked() as state:
            state["workers"][key] = dict(last_seen=time.time(), pid=os.getpid())
            advance(state)
            if state["status"] == "complete": return
            if (ROOT / "PAUSE").exists():
                atomic(ROOT / "status.json", status(state))
            else:
                eligible = [(k,t) for k,t in state["tasks"].items() if t["status"] == "pending" and
                            (ROOT / "data" / t["spec"]["dataset"] / "manifest.json").exists() and
                            (ROOT / "models" / f'{t["spec"]["model"]}.json').exists()]
                eligible.sort(key=lambda x:(priorities[x[1]["spec"]["stage"]],
                                             GROUPS.index((x[1]["spec"]["model"],x[1]["spec"]["dataset"])), x[0]))
                if eligible:
                    ident, task = eligible[0]
                    attempt = len(task["attempts"]) + 1
                    out = ROOT / "runs" / ident / f"attempt{attempt}"
                    out.mkdir(parents=True, exist_ok=True)
                    task["status"] = "running"
                    task["worker"] = key
                    task["attempts"].append(dict(started_at=time.time(), output=str(out), worker=key))
                    selected = (ident, task["path"], out)
                atomic(ROOT / "status.json", status(state))
        if selected is None:
            time.sleep(30)
            continue
        ident, spec, out = selected
        print("START", ident, flush=True)
        with (out / "train.log").open("w") as log:
            process = subprocess.run([PYTHON, "-u", str(ROOT/"train.py"), "--spec", spec, "--output", str(out)],
                                     stdout=log, stderr=subprocess.STDOUT)
        with locked() as state:
            task = state["tasks"][ident]
            task["attempts"][-1].update(finished_at=time.time(), returncode=process.returncode)
            if process.returncode == 0 and (out/"result.json").exists():
                task["result"] = str(out/"result.json")
                result(task)
                task["status"] = "done"
                print("DONE",ident, flush=True)
            else:
                # Preserve failed logs. One retry; repeated failures stop only this task.
                task["status"] = "pending" if len(task["attempts"]) < 2 else "failed"
                print("FAILED",ident,process.returncode,flush=True)
            advance(state)
            atomic(ROOT / "status.json", status(state))
    print("Worker reached 70-hour window; queue persists for another allocation.", flush=True)


def launch():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible or visible == [""]:
        raise RuntimeError("Missing Slurm GPU allocation")
    children = []
    for i,device in enumerate(visible):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
        log = (ROOT / "logs" / f'worker_{os.environ.get("SLURM_JOB_ID")}_{i}.log').open("a")
        p = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "worker", "--id", str(i)],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((p,log))
    failures = 0
    for p,log in children:
        failures += (p.wait() != 0)
        log.close()
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["init","status","worker","launch"])
    p.add_argument("--id", default="0")
    a = p.parse_args()
    if a.command == "init": initialize()
    elif a.command == "worker": worker(a.id)
    elif a.command == "launch": launch()
    else:
        with locked() as state: print(json.dumps(status(state),indent=2))
