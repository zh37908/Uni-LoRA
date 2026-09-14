"""Summarize P1 vision results (results/p1_vision/<dataset>_<size>/<method>_s<seed>.json).

<method> is lora / unilora / prolosa_r<ratio> (e.g. prolosa_r12to1).
Usage: python summarize_p1_vision.py [--result_root results/p1_vision] [--metric best_accuracy]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean, stdev

FILE_RE = re.compile(r"^(?P<method>lora|unilora|prolosa(?:_r\w+)?)_s(?:eed)?(?P<seed>\d+)\.json$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/p1_vision")
    parser.add_argument("--metric", default="best_accuracy", choices=["best_accuracy", "final_accuracy"])
    args = parser.parse_args()

    # results[(dataset_size, method)] -> list of acc
    results = defaultdict(list)
    for group in sorted(os.listdir(args.result_root)):
        group_dir = os.path.join(args.result_root, group)
        if not os.path.isdir(group_dir):
            continue
        for fname in sorted(os.listdir(group_dir)):
            m = FILE_RE.match(fname)
            if not m:
                continue
            with open(os.path.join(group_dir, fname), encoding="utf8") as f:
                record = json.load(f)
            results[(group, m.group("method"))].append(record[args.metric])

    print(f"{'dataset_size':<20} {'method':<16} {args.metric}")
    print("-" * 60)
    for (group, method), accs in sorted(results.items()):
        if len(accs) == 1:
            cell = f"{100 * accs[0]:.2f} (n=1)"
        else:
            cell = f"{100 * mean(accs):.2f} +/- {100 * stdev(accs):.2f} (n={len(accs)})"
        print(f"{group:<20} {method:<16} {cell}")


if __name__ == "__main__":
    main()
