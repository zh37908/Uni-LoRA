"""Submit the fixed grid in bounded waves respecting the 5880 queue limits."""
import json
import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TRACK=ROOT/'results/prolosa_grid_1m/dispatch.json'
PYTHON='/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python'
# Representative timing settings first, then the rest of the timing grid.
ORDER=[0,3,5,8,10,6,1,2,4,7,9]+list(range(11,25))
MAX_OUTSTANDING=6


def save(data):
    temp=TRACK.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(TRACK)


def refresh_summary():
    subprocess.run([PYTHON,'prolosa_grid_1m.py','summarize'],cwd=ROOT,check=True)


def main():
    TRACK.parent.mkdir(parents=True,exist_ok=True)
    lock=TRACK.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if TRACK.exists():
        state=json.loads(TRACK.read_text())
        state['controller_job']=os.environ.get('SLURM_JOB_ID')
    else:
        state=dict(started_epoch=time.time(),controller_job=os.environ.get('SLURM_JOB_ID'),
                   priority_order=ORDER,max_outstanding=MAX_OUTSTANDING,jobs={},snapshots=[])
    save(state)
    deadline=state['started_epoch']+71.5*3600
    previous_finished=-1;last_summary=0
    while time.time()<deadline:
        # Fail closed: a scheduler-query failure must not trigger extra submissions.
        try:
            listing=subprocess.check_output(
                ['squeue','--user=hzhaobi','--partition=gpu-rtx5880','--array','--noheader','--format=%i'],
                text=True,timeout=30).splitlines()
        except (subprocess.SubprocessError,OSError) as error:
            print('Queue query failed; retry without submitting:',error,flush=True)
            time.sleep(60);continue
        active={x.strip().split('_')[0] for x in listing}
        outstanding=sum(job['job_id'] in active for job in state['jobs'].values())
        submitted=len(state['jobs']);finished=submitted-outstanding
        # Leave one slot below MaxSubmitPU=10, including the three earlier baselines.
        slots=min(MAX_OUTSTANDING-outstanding, max(0,9-len(listing)))
        for task_id in (i for i in ORDER if str(i) not in state['jobs']):
            if slots<=0:break
            try:
                out=subprocess.check_output(
                    ['sbatch','--parsable',f'--array={task_id}','submit_prolosa_grid_1m_1gpu.sh'],
                    cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=30)
                job_id=out.strip().splitlines()[-1].split(';')[0]
                if not job_id.isdigit():raise ValueError(out)
            except (subprocess.SubprocessError,ValueError) as error:
                print('Submission paused:',error,flush=True);break
            state['jobs'][str(task_id)]=dict(job_id=job_id,submitted_epoch=time.time())
            save(state);slots-=1
            print(f'Submitted grid_{task_id:02d}: {job_id}_{task_id}',flush=True)
        now=time.time()
        if finished!=previous_finished or now-last_summary>=3600:
            refresh_summary();previous_finished=finished;last_summary=now
        for hour in (24,48,71):
            if now-state['started_epoch']>=hour*3600 and hour not in state['snapshots']:
                refresh_summary()
                for suffix in ('md','json'):
                    shutil.copyfile(TRACK.parent/f'summary.{suffix}',TRACK.parent/f'summary_{hour}h.{suffix}')
                state['snapshots'].append(hour);save(state)
        if len(state['jobs'])==len(ORDER) and all(j['job_id'] not in active for j in state['jobs'].values()):
            # Jobs just submitted are not present in the older listing; wait one cycle.
            if all(now-j['submitted_epoch']>90 for j in state['jobs'].values()):
                state['dispatcher_status']='all submitted jobs left the queue'
                save(state);refresh_summary();return
        time.sleep(60)
    state['dispatcher_status']='71.5-hour reporting window ended; inspect remaining jobs'
    state['unsubmitted']=[i for i in ORDER if str(i) not in state['jobs']]
    save(state);refresh_summary()


if __name__=='__main__':main()
