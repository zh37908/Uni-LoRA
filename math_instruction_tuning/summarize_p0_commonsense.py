"""Summarize P0 commonsense results into a mean +/- std table (8 benchmarks + avg).

Usage: python summarize_p0_commonsense.py [--result_root results/p0_commonsense]
"""

import argparse
import os
import re
from collections import defaultdict
from statistics import mean, stdev

DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
]
ACC_RE = re.compile(r"acc====\s*([0-9.]+)")
RUN_RE = re.compile(r"^(?P<model>.+?)_(?P<method>lora|unilora|prolosa(?:_r\w+)?)_s(?:eed)?(?P<seed>\d+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/p0_commonsense")
    args = parser.parse_args()

    # results[(model, method)][dataset] -> list of acc across seeds
    results = defaultdict(lambda: defaultdict(list))
    for run_name in sorted(os.listdir(args.result_root)):
        m = RUN_RE.match(run_name)
        if not m:
            continue
        run_dir = os.path.join(args.result_root, run_name)
        for dataset in DATASETS:
            log_path = os.path.join(run_dir, f"{dataset}.log")
            if not os.path.isfile(log_path):
                continue
            with open(log_path, encoding="utf8", errors="replace") as f:
                matches = ACC_RE.findall(f.read())
            if matches:
                results[(m.group("model"), m.group("method"))][dataset].append(float(matches[-1]))

    header = f"{'model':<10} {'method':<16}" + "".join(f" {d[:9]:>16}" for d in DATASETS) + f" {'avg':>16}"
    print(header)
    print("-" * len(header))
    for (model, method), ds_map in sorted(results.items()):
        row = f"{model:<10} {method:<16}"
        per_seed_means = []
        for dataset in DATASETS:
            accs = ds_map.get(dataset, [])
            if not accs:
                row += f" {'-':>16}"
            elif len(accs) == 1:
                row += f" {100 * accs[0]:>13.2f}(1)"
            else:
                row += f" {100 * mean(accs):>7.2f}+/-{100 * stdev(accs):<4.2f}({len(accs)})"
        # average across datasets of per-dataset seed means (only when all present)
        if all(ds_map.get(d) for d in DATASETS):
            avg = mean(mean(ds_map[d]) for d in DATASETS)
            row += f" {100 * avg:>16.2f}"
        else:
            row += f" {'-':>16}"
        print(row)


if __name__ == "__main__":
    main()
