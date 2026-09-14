"""Summarize E1/E2 theory-validation results and the compression gap Delta.

Delta(x) = Acc_method(x) - Acc_lora(x) computed on seed means, printed for
Uni-LoRA and every ProLoSA ratio variant (theory predicts Delta decreases with
sample size and increases with label noise).

Result files: <result_root>/<frac|eta><x>/<method>_s<seed>.json, where
<method> is lora / unilora / prolosa_r<ratio> (e.g. prolosa_r12to1).

Usage:
    python summarize_theory_results.py --result_root results/e1_sample_sweep --sweep frac
    python summarize_theory_results.py --result_root results/e2_label_noise --sweep eta
"""

import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean, stdev

FILE_RE = re.compile(r"^(?P<method>lora|unilora|prolosa(?:_r\w+)?)_s(?:eed)?(?P<seed>\d+)\.json$")


def method_sort_key(method):
    order = {"lora": 0, "unilora": 1}
    return (order.get(method, 2), method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--sweep", choices=["frac", "eta"], required=True)
    args = parser.parse_args()

    prefix = args.sweep
    # results[x_value][method] -> list of acc
    results = defaultdict(lambda: defaultdict(list))
    for group in os.listdir(args.result_root):
        if not group.startswith(prefix):
            continue
        x = float(group[len(prefix):])
        group_dir = os.path.join(args.result_root, group)
        for fname in os.listdir(group_dir):
            m = FILE_RE.match(fname)
            if not m:
                continue
            with open(os.path.join(group_dir, fname), encoding="utf8") as f:
                record = json.load(f)
            results[x][m.group("method")].append(record["accuracy"])

    methods = sorted({m for by_method in results.values() for m in by_method}, key=method_sort_key)
    delta_methods = [m for m in methods if m != "lora"]
    header = f"{prefix:>8}" + "".join(f" {m:>23}" for m in methods)
    header += "".join(f" {'D(' + m + ')':>18}" for m in delta_methods)
    print(header)
    print("-" * len(header))
    for x in sorted(results):
        row = f"{x:>8}"
        means = {}
        for method in methods:
            accs = results[x].get(method, [])
            if not accs:
                row += f" {'-':>23}"
                continue
            means[method] = mean(accs)
            std = stdev(accs) if len(accs) > 1 else 0.0
            row += f" {100 * means[method]:>12.2f}+/-{100 * std:<5.2f}({len(accs)})"[:24].rjust(24)
        for method in delta_methods:
            if "lora" in means and method in means:
                row += f" {100 * (means[method] - means['lora']):>+18.2f}"
            else:
                row += f" {'-':>18}"
        print(row)


if __name__ == "__main__":
    main()
