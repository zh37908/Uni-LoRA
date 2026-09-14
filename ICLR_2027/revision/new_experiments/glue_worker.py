from pathlib import Path
import argparse,json,subprocess,sys,time
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('worker',type=int);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
paths=json.loads(Path(args.manifest).read_text());start=time.time()
status=P/f'{Path(args.manifest).stem}_worker_{args.worker}.json'
for index in range(args.worker,len(paths),args.workers):
    path=paths[index];spec=json.loads(Path(path).read_text());out=P/'results'/spec['id'];out.mkdir(parents=True,exist_ok=True)
    state=dict(manifest=args.manifest,worker=args.worker,index=index,total=len(paths),active_id=spec['id'],elapsed=time.time()-start)
    status.write_text(json.dumps(state,indent=2)+'\n')
    print('START',spec['id'],flush=True)
    with (out/'launch.log').open('a') as log:
        run=subprocess.run(['/home/hzhaobi/miniconda3/envs/unilora_nlu/bin/python',str(P/'run_glue.py'),'--spec',path],stdout=log,stderr=subprocess.STDOUT,cwd=P.parents[2])
    if run.returncode:
        state.update(state='failed',returncode=run.returncode);status.write_text(json.dumps(state,indent=2)+'\n');sys.exit(run.returncode)
    print('DONE',spec['id'],flush=True)
status.write_text(json.dumps(dict(state='complete',worker=args.worker,elapsed=time.time()-start),indent=2)+'\n')
