"""Submit a bounded seed wave using site-supplied Slurm account and QoS."""
import argparse,json,os,subprocess
from pathlib import Path
from common import ROOT,STORAGE

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--account',required=True);p.add_argument('--partition',default='normal');p.add_argument('--qos',default='normal_qos')
    p.add_argument('--seeds',required=True,help='Comma-separated subset of 0,21,42; e.g. 0,21 then 42 after completion')
    p.add_argument('--max-active',type=int,default=8);a=p.parse_args()
    seeds={int(s) for s in a.seeds.split(',')};assert seeds and seeds<={0,21,42}
    configs=[f for f in sorted((ROOT/'configs').glob('*.json')) if json.loads(f.read_text())['seed'] in seeds]
    configs=[f for f in configs if not (STORAGE/'runs/llama31_smol'/f.stem/'evaluation/metrics.json').exists()]
    # Count every live/pending job for this user, conservatively across QoS.
    jobs=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i'],text=True).splitlines()
    if len(jobs)+len(configs)>a.max_active:raise SystemExit('This wave exceeds --max-active; wait for the previous wave or submit fewer seeds.')
    env=os.environ.copy();env['UNILORA_PACKAGE']=str(ROOT)
    for config in configs:
        job=subprocess.check_output(['sbatch','--parsable',f'--account={a.account}',f'--partition={a.partition}',f'--qos={a.qos}',f'--job-name={config.stem}',f'--output={STORAGE}/logs/%x_%j.out',f'--error={STORAGE}/logs/%x_%j.err',str(ROOT/'run.sbatch'),str(config)],env=env,text=True).strip()
        with (STORAGE/'logs/submissions.jsonl').open('a') as f:f.write(json.dumps({'config':config.name,'job':job})+'\n')
        print(config.name,job,flush=True)

if __name__=='__main__':main()
