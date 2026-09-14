"""Lock hyperparameters using training-holdout scores only; submit final runs."""
from pathlib import Path
import json,math,statistics,subprocess
P=Path(__file__).resolve().parent;manifest=json.loads((P/'tune_manifest.json').read_text())
groups={}
for path in manifest:
    s=json.loads(Path(path).read_text());r=json.loads((P/'results'/s['id']/'result.json').read_text())
    assert r['spec']==s and 'heldout_score' not in r and s['phase']=='tune'
    assert math.isfinite(r['selection_score']) and math.isfinite(r['selection_loss'])
    groups.setdefault((s['task'],s['method'],s['config_id']),[]).append(r)
locked={};paths=[]
for task in ['cola','mrpc']:
    for method in ['lora','unilora','prolosa','rosa']:
        candidates=[]
        for cid in range(12):
            runs=groups[task,method,cid];assert {r['spec']['seed'] for r in runs}=={101,102}
            candidates.append((statistics.mean(r['selection_score'] for r in runs),-statistics.mean(r['selection_loss'] for r in runs),-cid,runs))
        score,_,_,runs=max(candidates,key=lambda x:x[:3]);s=runs[0]['spec'].copy()
        locked[f'{task}/{method}']=dict(config_id=s['config_id'],mean_selection_score=score,tuning_seeds=[101,102],spec=s)
        for seed in [201,202,203,204,205]:
            z=s.copy();z.update(id=f'final_{task}_{method}_s{seed}',phase='final',seed=seed)
            path=P/'specs'/(z['id']+'.json');path.write_text(json.dumps(z,indent=2)+'\n');paths.append(str(path))
lock_path=P/'locked_selection.json'
if lock_path.exists():assert json.loads(lock_path.read_text())==locked
else:lock_path.write_text(json.dumps(locked,indent=2)+'\n')
(P/'final_manifest.json').write_text(json.dumps(paths,indent=2)+'\n')
receipt=P/'final_jobs.json'
submitted=json.loads(receipt.read_text()) if receipt.exists() else {}
if 'final_job' not in submitted:
    job=subprocess.check_output(['sbatch','--parsable','--array=0-1',str(P/'glue_worker.sh'),str(P/'final_manifest.json')],text=True).strip()
    submitted['final_job']=job;receipt.write_text(json.dumps(submitted,indent=2)+'\n')
if 'summary_job' not in submitted:
    summary=subprocess.check_output(['sbatch','--parsable','--dependency=afterok:'+submitted['final_job'],str(P/'summary_job.sh'),'glue'],text=True).strip()
    submitted['summary_job']=summary;receipt.write_text(json.dumps(submitted,indent=2)+'\n')
print('Final pipeline:',submitted)
