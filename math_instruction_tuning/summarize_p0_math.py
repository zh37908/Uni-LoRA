"""Summarize P0 math results (GSM8K / MATH500 / MATH) into a mean +/- std table.

Parses the acc lines printed by instruction_tuning_eval/{gsm8k_eval,MATH_eval}.py
from results/p0_math/<model>_<method>_s<seed>/{gsm8k,math500,math}.log, where
<method> is lora / unilora / prolosa_r<ratio> (e.g. prolosa_r12to1).

Usage: python summarize_p0_math.py [--result_root results/p0_math]
"""

import argparse
import os
import re
from collections import defaultdict
from statistics import mean, stdev

ACC_PATTERNS = {
    "gsm8k.log": re.compile(r"gsm8k acc====\s*([0-9.]+)"),
    "math500.log": re.compile(r"acc====\s*([0-9.]+)"),
    "math.log": re.compile(r"acc====\s*([0-9.]+)"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/p0_math")
    args = parser.parse_args()

    # results[(model, method)][benchmark] -> list of acc across seeds
    results = defaultdict(lambda: defaultdict(list))
    run_re = re.compile(r"^(?P<model>.+?)_(?P<method>lora|unilora|prolosa(?:_r\w+)?)_s(?:eed)?(?P<seed>\d+)$")

    for run_name in sorted(os.listdir(args.result_root)):
        m = run_re.match(run_name)
        if not m:
            continue
        run_dir = os.path.join(args.result_root, run_name)
        for log_name, pattern in ACC_PATTERNS.items():
            log_path = os.path.join(run_dir, log_name)
            if not os.path.isfile(log_path):
                continue
            with open(log_path, encoding="utf8", errors="replace") as f:
                matches = pattern.findall(f.read())
            if matches:
                bench = log_name.removesuffix(".log")
                results[(m.group("model"), m.group("method"))][bench].append(float(matches[-1]))

    benches = ["gsm8k", "math500", "math"]
    header = f"{'model':<10} {'method':<16}" + "".join(f" {b:>22}" for b in benches)
    print(header)
    print("-" * len(header))
    for (model, method), bench_map in sorted(results.items()):
        row = f"{model:<10} {method:<16}"
        for bench in benches:
            accs = bench_map.get(bench, [])
            if not accs:
                cell = "-"
            elif len(accs) == 1:
                cell = f"{100 * accs[0]:.2f} (n=1)"
            else:
                cell = f"{100 * mean(accs):.2f} +/- {100 * stdev(accs):.2f} (n={len(accs)})"
            row += f" {cell:>22}"
        print(row)


if __name__ == "__main__":
    main()
