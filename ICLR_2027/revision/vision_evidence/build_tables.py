"""Regenerate paper rows and an audited CSV from completed local ViT runs.

Run from any directory: python ICLR_2027/revision/vision_evidence/build_tables.py
Uses only the standard library; does not train models or modify source results.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SETTINGS = ("cifar100_1000", "food101_1000", "dtd_1000", "cifar100_5000", "cifar100_full")
METHODS = ("lora", "unilora", "prolosa_r2to1", "prolosa_r4to1", "prolosa_r8to1", "prolosa_r12to1")
LABELS = (r"LoRA ($r=4$)", "Uni-LoRA", r"ProLoSA ($2:1$)", r"ProLoSA ($4:1$)", r"ProLoSA ($8:1$)", r"ProLoSA ($12:1$)")


def field(text, name):
    match = re.search(r"^" + re.escape(name) + r"=(.*),$", text, re.M)
    if match is None:
        raise ValueError(f"Missing logged argument: {name}")
    return match[1]


def main():
    logs = defaultdict(list)
    for path in sorted((ROOT / "ViT/logs").glob("*.out")):
        text = path.read_text(errors="replace")
        result = re.search(r"^RESULT: (.+)$", text, re.M)
        source = re.search(r"result_file='([^']+)'", text)
        if result and source:
            logs[source[1]].append((path, text, json.loads(result[1])))

    records = []
    for runset, budget in (("p1_vision", 24600), ("p1_vision_d72k", 72000)):
        indexed = {}
        for path in sorted((ROOT / "ViT/results" / runset).glob("*/*.json")):
            raw = json.loads(path.read_text())
            candidates = [x for x in logs[str(path.relative_to(ROOT / "ViT"))] if x[2] == raw]
            if len(candidates) != 1:
                raise ValueError(f"Expected one matching completed log: {path}")
            log, text, _ = candidates[0]
            method = path.stem.rsplit("_s", 1)[0]
            sparse = 0
            latent = 0 if method == "lora" else int(field(text, "vector_length"))
            if method.startswith("prolosa"):
                latent = int(field(text, "theta_d_length"))
                sparse = int(field(text, "rosa_sparse_budget"))
                activated = re.search(r"Activated .*selected_positions=(\d+)", text)
                assert activated and int(activated[1]) == sparse, path
                assert raw["adapter_params"] == latent, path
                assert latent + sparse == budget, path
            else:
                assert raw["adapter_params"] == (147456 if method == "lora" else budget), path
            head = raw["trainable_params"] - raw["adapter_params"]
            assert head == 769 * {"cifar100": 100, "food101": 101, "dtd": 47}[raw["dataset"]], path
            assert raw["subset_seed"] == 42 and int(field(text, "lora_r")) == 4, path
            active = 147456 if method == "lora" else latent + sparse
            record = dict(
                runset=runset, setting=path.parent.name, method=method,
                seed=raw["seed"], subset_seed=raw["subset_seed"],
                final_accuracy_pct=100 * raw["final_accuracy"],
                best_accuracy_pct=100 * raw["best_accuracy"],
                latent_d=latent, sparse_K=sparse, adapter_active_dof=active,
                head_params=head, total_active_dof=active + head,
                raw_adapter_params=raw["adapter_params"],
                raw_trainable_params=raw["trainable_params"],
                head_lr=re.search(r"head_lr=([\d.e+-]+)", text)[1],
                source_json=str(path.relative_to(ROOT)), source_log=str(log.relative_to(ROOT)),
            )
            for name in ("learning_rate", "learning_rate_theta_d", "num_train_epochs",
                         "per_device_train_batch_size", "gradient_accumulation_steps",
                         "rosa_warmup_steps", "rosa_mask_steps", "rosa_sparse_lr_mult",
                         "warmup_ratio", "lr_scheduler_type", "optim", "bf16"):
                record[name] = field(text, name)
            records.append(record)
            indexed[(path.parent.name, method, raw["seed"])] = record

        rows = []
        for method, label in zip(METHODS, LABELS):
            values = [indexed[(setting, method, 42)] for setting in SETTINGS]
            cells = [label] + [f'{r["final_accuracy_pct"]:.2f}' for r in values]
            rows.append(" & ".join(cells) + r" \\")
        header = r"""\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{3}{c}{1,000 training images} & \multicolumn{2}{c}{CIFAR-100} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Method & CIFAR-100 & Food-101 & DTD & 5,000 & Full \\
\midrule
"""
        footer = "\\bottomrule\n\\end{tabular}\n"
        (OUT / f"{runset}_rows.tex").write_text("% Generated from final_accuracy, seed 42; do not edit.\n" + header + "\n".join(rows) + "\n" + footer)

        allocations = []
        for method, label in zip(METHODS[1:], LABELS[1:]):
            r = indexed[(SETTINGS[0], method, 42)]
            allocations.append(f'{label} & {r["latent_d"]:,} & {r["sparse_K"]:,}' + r" \\")
        (OUT / f"{runset}_allocations.tex").write_text("% Generated exact active budgets.\n" + "\n".join(allocations) + "\n")

    with (OUT / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Audited {len(records)} completed runs against logs; generated four TeX fragments and runs.csv.")


if __name__ == "__main__":
    main()
