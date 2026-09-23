"""Summarize all four fixed configurations, including the backup, over three seeds."""
import argparse,csv,json,math,statistics
from pathlib import Path
from common import ROOT,sha256
GROUPS=['lora','unilora','prolosa_primary','prolosa_backup']

def summarize(rows):
    result={}
    for group in GROUPS:
        selected=sorted([r for r in rows if r['group']==group],key=lambda r:int(r['seed']))
        assert [int(r['seed']) for r in selected]==[0,21,42],f'Incomplete/duplicate seeds for {group}'
        result[group]={}
        for key in ['ifeval_prompt_strict','ifbench_prompt_loose']:
            values=[float(r[key]) for r in selected]
            assert all(math.isfinite(v) and 0<=v<=100 for v in values)
            result[group][key]={'mean_percent':statistics.mean(values),'sample_sd_pp':statistics.stdev(values),'new_seeds_0_21_mean_percent':statistics.mean(values[:2]),'per_seed_percent':dict(zip(['0','21','42'],values))}
    return result

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--recorded',action='store_true');g.add_argument('--runs',type=Path);a=p.parse_args()
    if a.recorded:
        with (ROOT/'results/three_seeds.csv').open() as f:rows=list(csv.DictReader(f))
    else:
        rows=[]
        for cfg in sorted((ROOT/'configs').glob('*.json')):
            c=json.loads(cfg.read_text());run=a.runs/cfg.stem
            train=json.loads((run/'metrics.json').read_text());ev=json.loads((run/'evaluation/metrics.json').read_text())
            manifest=json.loads((run/'run_manifest.json').read_text())
            assert train['steps']==1563 and math.isfinite(train['validation']['nll'])
            for k,v in c.items():assert manifest['config'][k]==v,(cfg.name,k)
            for b,n in [('ifeval',541),('ifbench',300)]:
                f=run/f'evaluation/{b}_predictions.jsonl'
                assert len(f.read_text().splitlines())==n==ev['benchmarks'][b]['n']
                assert sha256(f)==ev['benchmarks'][b]['predictions_sha256']
            rows.append({'group':cfg.stem.rsplit('_r4_seed',1)[0],'seed':c['seed'],'ifeval_prompt_strict':100*ev['benchmarks']['ifeval']['prompt_strict'],'ifbench_prompt_loose':100*ev['benchmarks']['ifbench']['prompt_loose']})
    print(json.dumps(summarize(rows),indent=2))

if __name__=='__main__':main()
