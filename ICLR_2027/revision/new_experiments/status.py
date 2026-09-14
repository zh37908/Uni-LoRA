"""Read-only progress for the revision experiments."""
from pathlib import Path
import json
import subprocess

P = Path(__file__).resolve().parent
jobs = json.loads((P/'jobs.json').read_text())
for phase, expected in [('tune', 192), ('final', 40)]:
    results = list((P/'results').glob(phase+'_*/result.json'))
    print(f'GLUE {phase}: {len(results)}/{expected} complete')
root = P.parents[2]/'synthetic/results_revision5_fixed_teacher/trials_v2'
print(f'Synthetic trials: {len(list(root.glob("*/result.json")))}/480 complete')
if (P/'final_jobs.json').exists():
    jobs.update(json.loads((P/'final_jobs.json').read_text()))
ids = []
for key, value in jobs.items():
    if key.startswith('superseded'):
        continue
    ids.extend(value if isinstance(value, list) else [value])
subprocess.run(['squeue', '-j', ','.join(ids), '-o', '%.18i %.24j %.10T %.10M %R'])
for phase in ['glue', 'synthetic']:
    report = P/f'summary_{phase}'/'README.md'
    if report.exists():
        print('Summary available:', report)
