"""After an allocation: export all 80 results, or resume an unfinished queue once."""
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import pipeline as q

ORDER = ["oxfordpets", "stanfordcars", "cifar10", "dtd", "eurosat", "fgvc", "resisc45", "cifar100"]


def export(state):
    tasks = q.tasks_for(state,"final")
    assert len(tasks)==80 and q.complete(tasks)
    records = []
    for task in tasks:
        r = q.result(task)
        assert r["spec"]["seed"] in range(42,47)
        assert r["spec"]["epochs"] == 20
        records.append(dict(model=r["spec"]["model"], dataset=r["spec"]["dataset"],
                            seed=r["spec"]["seed"], test_accuracy_pct=100*r["test_accuracy"],
                            validation_accuracy_pct=100*r["best_val_accuracy"],
                            best_epoch=r["best_epoch"], d=r["d"], K=r["K"],
                            vector_lr=r["spec"]["vector_lr"], head_lr=r["spec"]["head_lr"],
                            source=task["result"]))
    with (q.ROOT/"final_runs.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    tex=[]; summary={}
    for model in ("base","large"):
        cells=[]; per_seed={s:[] for s in range(42,47)}
        for dataset in ORDER:
            rows=[r for r in records if r["model"]==model and r["dataset"]==dataset]
            assert len(rows)==5 and {r["seed"] for r in rows}==set(range(42,47))
            scores=[r["test_accuracy_pct"] for r in rows]
            mean,std=statistics.mean(scores),statistics.stdev(scores)
            cells.append(f"${mean:.2f}_{{\\pm {std:.2f}}}$")
            for r in rows: per_seed[r["seed"]].append(r["test_accuracy_pct"])
            summary[f"{model}/{dataset}"]=dict(mean=mean,std=std)
        seed_avgs=[statistics.mean(v) for v in per_seed.values()]
        summary[f"{model}/average"]=dict(mean=statistics.mean(seed_avgs),std=statistics.stdev(seed_avgs),seed_average=seed_avgs)
        budget="72K" if model=="base" else "144K"
        tex.append(" & ".join(["", "ProLoSA",budget,*cells,f"{statistics.mean(seed_avgs):.2f}"])+r" \\")
    q.atomic(q.ROOT/"table5_statistics.json",summary)
    (q.ROOT/"table5_prolosa_rows.tex").write_text("% Generated only after all 80 held-out-test runs completed.\n"+"\n".join(tex)+"\n")
    notes = ["# Completed ProLoSA Table 5 experiments", "",
             "80 formal runs: 2 backbones × 8 datasets × seeds 42–46; 20 epochs.",
             "Configuration selection used only validation data, with tuning seeds 101/102.",
             "Each formal checkpoint was selected on validation before evaluating test.",
             "Table cells report test accuracy mean ± sample standard deviation.",
             "The eight-task average gives equal weight to each dataset.",
             "Historical baselines must be identified as quoted results; splits were reconstructed and fixed as documented in README.md.",
             "", "| Backbone | Eight-task average |", "|---|---:|"]
    for model in ("base","large"):
        avg=summary[f"{model}/average"]
        notes.append(f'| ViT-{model} | {avg["mean"]:.2f} ± {avg["std"]:.2f} |')
    (q.ROOT/"RESULTS.md").write_text("\n".join(notes)+"\n")
    print("EXPORTED 80 completed formal runs",flush=True)


def main():
    with q.locked() as state:
        if state["status"]=="complete":
            export(state)
            return
        failed=[k for k,t in state["tasks"].items() if t["status"]=="failed"]
        missing=[d for d in q.DATASETS if not (q.ROOT/"data"/d/"manifest.json").exists()]
        if failed or missing:
            q.atomic(q.ROOT/"attention_required.json",dict(failed_tasks=failed,missing_datasets=missing,at=time.time()))
            raise RuntimeError("Cannot automatically continue: failed trials or missing data; inspect attention_required.json")
        if state.get("renewals",0)>=1:
            raise RuntimeError("Automatic continuation limit reached; queue preserved for inspection")
        # This script is scheduled afterany on the full prior allocation, so its workers are gone.
        for t in state["tasks"].values():
            if t["status"]=="running":
                t["status"]="pending"
                t["attempts"][-1]["interrupted_allocation"]=True
        state["renewals"]=state.get("renewals",0)+1
    output=subprocess.check_output(["sbatch","--parsable",str(q.ROOT/"submit.sh")],text=True).strip()
    job=output.split(";")[0]
    assert job.isdigit(),output
    continuation=subprocess.check_output(["sbatch","--parsable",f"--dependency=afterany:{job}",str(q.ROOT/"finish.sh")],text=True).strip()
    with q.locked() as state:
        state.setdefault("allocations",[]).append(dict(job_id=job,finisher=continuation,submitted_at=time.time()))
    print("CONTINUED",job,"FINISHER",continuation,flush=True)


if __name__=="__main__":main()
